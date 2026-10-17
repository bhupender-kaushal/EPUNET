import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import numpy as np

# --- 1. EVIDENTIAL LOSS COMPONENTS ---



def dirichlet_kl_divergence(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c), dtype=torch.float32, device=alpha.device)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    
    return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

def stochastic_ohem_mask(loss_map, k_percent=0.02, temperature=1.0):
    """Generates a binary mask for the top-k hardest pixels using Gumbel-Softmax."""
    B = loss_map.size(0)
    loss_flat = loss_map.view(B, -1)
    k_num = max(1, int(k_percent * loss_flat.size(1)))
    
    logits = torch.log(loss_flat + 1e-8)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    perturbed_logits = (logits + gumbel_noise) / temperature
    
    _, indices = torch.topk(perturbed_logits, k_num, dim=-1)
    mask = torch.zeros_like(loss_flat).scatter_(-1, indices, 1.0)
    
    return mask, k_num

def unreduced_evidential_loss(logits, mask, c=2, eps=1e-10):
    """Returns unreduced spatial maps for pixel-wise masking."""
    B, C, H, W = logits.shape
    
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    
    alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, c)
    mask_flat = mask.view(-1)
    
    S = torch.sum(alpha_flat, dim=1, keepdim=True)
    E = alpha_flat - 1.0
    expected_p = alpha_flat / S
    
    label = F.one_hot(mask_flat, num_classes=c).float()
    
    # 1. Expected Cross Entropy Map [B*H*W, 1]
    L_ace_map = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha_flat)), dim=1, keepdim=True)
    
    # 2. Expected Dice Loss (Remains Global)
    p_tumor = expected_p[:, 1]
    y_tumor = label[:, 1]
    smooth = 1e-5
    intersection = torch.sum(p_tumor * y_tumor)
    cardinality = torch.sum(p_tumor) + torch.sum(y_tumor)
    L_e_dice = 1.0 - ((2.0 * intersection + smooth) / (cardinality + smooth))
    
    # 3. Dirichlet KL Penalty Map [B*H*W, 1]
    alp_false = E * (1.0 - label) + 1.0
    L_KL_map = dirichlet_kl_divergence(alp_false, c)
    
    # 4. Accuracy-Uncertainty Calibration Map [B*H*W, 1]
    pred_scores, pred_cls = torch.max(expected_p, 1, keepdim=True)
    uncertainty = c / S
    target = mask_flat.view(-1, 1).float()
    acc_match = torch.eq(pred_cls, target).float()
    
    acc_uncertain = -torch.log(pred_scores * (1.0 - uncertainty) + eps)
    inacc_certain = -torch.log((1.0 - pred_scores) * uncertainty + eps)
    L_AU_map = acc_match * acc_uncertain + (1.0 - acc_match) * inacc_certain
    
    # Reshape back to spatial dimensions
    L_ace_map = L_ace_map.view(B, H, W)
    L_KL_map = L_KL_map.view(B, H, W)
    L_AU_map = L_AU_map.view(B, H, W)
    
    return L_ace_map, L_e_dice, L_KL_map, L_AU_map

# --- 2. HIERARCHICAL ARCHITECTURE ---

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels=None):
        super(ResBlock, self).__init__()
        if down_channels is None: down_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, down_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(down_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)

class HierarchicalCore(nn.Module):
    def __init__(self, in_channels, latent_dims, channels_per_block):
        super(HierarchicalCore, self).__init__()
        self.latent_dims = latent_dims
        self.num_levels = len(channels_per_block)
        self.num_latent_levels = len(latent_dims)

        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        for out_channels in channels_per_block:
            self.encoder_blocks.append(
                nn.Sequential(*[ResBlock(curr_channels if j == 0 else out_channels, out_channels) for j in range(3)])
            )
            curr_channels = out_channels
            
        self.dist_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        rev_channels = channels_per_block[::-1]
        curr_dec_channels = rev_channels[0]
        
        for i, latent_dim in enumerate(latent_dims):
            self.dist_convs.append(nn.Conv2d(curr_dec_channels, 2 * latent_dim, kernel_size=1))
            next_skip_channels = rev_channels[i + 1]
            in_ch = curr_dec_channels + latent_dim + next_skip_channels
            out_ch = rev_channels[i + 1]
            self.decoder_blocks.append(nn.Sequential(*[ResBlock(in_ch if j==0 else out_ch, out_ch) for j in range(3)]))
            curr_dec_channels = out_ch

        for i, latent_dim in enumerate(latent_dims):
            dist_layer = nn.Conv2d(curr_dec_channels, 2 * latent_dim, kernel_size=1)
            
            # VAE Zero-Head Initialization
            torch.nn.init.zeros_(dist_layer.weight)
            torch.nn.init.zeros_(dist_layer.bias)
            
            self.dist_convs.append(dist_layer)

    def forward(self, inputs, use_mean=False, z_q=None):
        encoder_features = inputs
        encoder_outputs = []
        for i, block in enumerate(self.encoder_blocks):
            encoder_features = block(encoder_features)
            encoder_outputs.append(encoder_features)
            if i != self.num_levels - 1:
                encoder_features = F.avg_pool2d(encoder_features, kernel_size=2, stride=2)
                
        decoder_features = encoder_outputs[-1]
        distributions, used_latents = [], []
        rev_encoder_outputs = encoder_outputs[::-1]
        
        for i in range(self.num_latent_levels):
                latent_dim = self.latent_dims[i]
                mu_logsigma = self.dist_convs[i](decoder_features)
                

                mu = torch.clamp(mu_logsigma[:, :latent_dim, :, :], min=-50.0, max=50.0)
                logsigma = torch.clamp(mu_logsigma[:, latent_dim:, :, :], min=-10.0, max=2.0)
                

                scale = torch.exp(logsigma) + 1e-7
                
                dist = Independent(Normal(loc=mu, scale=scale), 1)
                distributions.append(dist)
                
                z = z_q[i] if z_q is not None else (dist.mean if use_mean else dist.rsample())
                used_latents.append(z)
                
                dec_out_lo = torch.cat([z, decoder_features], dim=1)
                dec_out_hi = F.interpolate(dec_out_lo, scale_factor=2, mode='bilinear', align_corners=False)
                
                # FIX: Apply Spatial Dropout to the Skip Connection
                skip_features = rev_encoder_outputs[i + 1]
                if self.training:
                    skip_features = F.dropout2d(skip_features, p=0.15)
                    
                decoder_features = torch.cat([dec_out_hi, skip_features], dim=1)
                decoder_features = self.decoder_blocks[i](decoder_features)
                
        return {'decoder_features': decoder_features, 'encoder_features': encoder_outputs, 
                'distributions': distributions, 'used_latents': used_latents}

class StitchingDecoder(nn.Module):
    def __init__(self, latent_dims, channels_per_block, num_classes):
        super(StitchingDecoder, self).__init__()
        start_level = len(latent_dims) + 1
        num_levels = len(channels_per_block)
        self.decoder_blocks = nn.ModuleList()
        rev_channels = channels_per_block[::-1]
        curr_dec_channels = rev_channels[start_level - 1]
        
        for level in range(start_level, num_levels):
            skip_channels = rev_channels[level]
            in_ch = curr_dec_channels + skip_channels
            out_ch = rev_channels[level]
            self.decoder_blocks.append(nn.Sequential(*[ResBlock(in_ch if j==0 else out_ch, out_ch) for j in range(3)]))
            curr_dec_channels = out_ch
            
        self.final_conv = nn.Conv2d(curr_dec_channels, num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
            rev_encoder_outputs = encoder_features[::-1]
            start_level = len(rev_encoder_outputs) - len(self.decoder_blocks)
            
            for i, block in enumerate(self.decoder_blocks):
                level_idx = start_level + i 
                
                decoder_features = F.interpolate(decoder_features, scale_factor=2, mode='bilinear', align_corners=False)
                
                # FIX: Apply Spatial Dropout to the Skip Connection
                skip_features = rev_encoder_outputs[level_idx]
                # if self.training:
                #     skip_features = F.dropout2d(skip_features, p=0.10)
                    
                decoder_features = torch.cat([decoder_features, skip_features], dim=1)
                decoder_features = block(decoder_features)
                
            return self.final_conv(decoder_features)

class HierarchicalEvidentialUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, latent_dims=[2, 2, 2, 2], channels_per_block=[24, 48, 96, 192, 192, 192, 192, 192]):
        super(HierarchicalEvidentialUNet, self).__init__()
        self.prior = HierarchicalCore(input_channels, latent_dims, channels_per_block)
        self.posterior = HierarchicalCore(input_channels + num_classes, latent_dims, channels_per_block)
        self.f_comb = StitchingDecoder(latent_dims, channels_per_block, num_classes)
        self.num_classes = num_classes

    def sample(self, image, use_mean=False):
        """Used during validation/testing to generate logits."""
        prior_out = self.prior(image, use_mean=use_mean)
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])

    def calculate_hierarchical_kl(self, prior_dists, posterior_dists):
        kl_sum = 0.0
        for p_dist, q_dist in zip(prior_dists, posterior_dists):
            kl_sum += torch.mean(torch.sum(kl.kl_divergence(q_dist, p_dist), dim=[1, 2]))
        return kl_sum


    def forward_train_step(self, image, mask_one_hot, mask_labels, lambda_geco, kappa, n_samples=3, current_epoch=0, total_epochs=500):
        prior_out = self.prior(image, use_mean=False)
        posterior_in = torch.cat([mask_one_hot, image], dim=1)
        posterior_out = self.posterior(posterior_in, use_mean=False)
        
        vae_kl_loss = self.calculate_hierarchical_kl(prior_out['distributions'], posterior_out['distributions'])
        
        total_masked_data_loss = 0.0
        
        # 1. PASS 1: STOCHASTIC (Data Fitting with Relaxed Dice for Variance)
        for _ in range(n_samples):
            z_sample = [dist.rsample() for dist in posterior_out['distributions']]
            prior_z_out = self.prior(image, z_q=z_sample)
            logits_stoch = self.f_comb(prior_z_out['encoder_features'], prior_z_out['decoder_features'])
            
            L_ace_map, L_e_dice_stoch, _, _ = unreduced_evidential_loss(logits_stoch, mask_labels, c=self.num_classes)
            
            ohem_mask, k_num = stochastic_ohem_mask(L_ace_map, k_percent=0.04) # 4% to prevent interior collapse
            B = L_ace_map.size(0)
            masked_ace = (L_ace_map.view(B, -1) * ohem_mask).sum() / (k_num * B)
            
            # RELAXED stochastic dice to allow the GED to drop
            lambda_dice_stoch = 0.2
            total_masked_data_loss += masked_ace + (lambda_dice_stoch * L_e_dice_stoch)
            
        mean_data_loss = total_masked_data_loss / n_samples
        
        # 2. PASS 2: DETERMINISTIC (Evidential Penalties & Strict Structural Anchor)
        z_mean = [dist.mean for dist in posterior_out['distributions']]
        prior_z_mean_out = self.prior(image, z_q=z_mean)
        logits_det = self.f_comb(prior_z_mean_out['encoder_features'], prior_z_mean_out['decoder_features'])
        
        _, L_e_dice_det, L_KL_map, L_AU_map = unreduced_evidential_loss(logits_det, mask_labels, c=self.num_classes)
        
        pen_mask, k_num_pen = stochastic_ohem_mask(L_KL_map + L_AU_map, k_percent=0.04)
        B = L_KL_map.size(0)
        masked_kl_edl = (L_KL_map.view(B, -1) * pen_mask).sum() / (k_num_pen * B)
        masked_au_edl = (L_AU_map.view(B, -1) * pen_mask).sum() / (k_num_pen * B)
        
        # --- DYNAMIC STRUCTURAL ANCHOR ---
        # Ramp from 0.5 to 1.5 over the first 100 epochs
        warmup_epochs = 100.0
        ramp_factor = min(1.0, current_epoch / warmup_epochs)
        lambda_dice_det = 0.5 + (1.0 * ramp_factor) 
        
        structural_anchor_loss = lambda_dice_det * L_e_dice_det        
        # 3. GECO Integration
        # target_kl = torch.tensor(20.0, device=image.device)
        # penalized_vae_kl = torch.abs(vae_kl_loss - target_kl)
        
        constraint_loss = lambda_geco * (mean_data_loss - kappa)
        
        # Final mathematical formulation: Add the structural anchor outside the GECO multiplier
        loss = constraint_loss + masked_kl_edl + masked_au_edl + vae_kl_loss + structural_anchor_loss        
        return loss, mean_data_loss, (masked_kl_edl + masked_au_edl), vae_kl_loss, L_e_dice_det
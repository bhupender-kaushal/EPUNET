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

def staged_hierarchical_evidential_loss(logits, mask, current_epoch, warmup_epochs=50, c=2, eps=1e-10):
    """
    Phase 1 (Epoch <= warmup_epochs): Expected CE only.
    Phase 2 (Epoch > warmup_epochs): Expected CE + Annealed Dirichlet KL + Annealed AU.
    """
    B, C, H, W = logits.shape
    
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    
    alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, c)
    mask_flat = mask.view(-1)
    
    S = torch.sum(alpha_flat, dim=1, keepdim=True)
    E = alpha_flat - 1.0
    expected_p = alpha_flat / S
    
    label = F.one_hot(mask_flat, num_classes=c).float()
    
    # 1. ALWAYS ACTIVE: Expected Cross Entropy (Data Fitting)
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha_flat)), dim=1, keepdim=True)
    
    # --- PHASE 1: WARMUP (Return Early) ---
    if current_epoch <= warmup_epochs:
        return torch.mean(L_ace)
        
    # --- PHASE 2: CALIBRATION ---
    active_epoch = current_epoch - warmup_epochs
    total_active_epochs = 200 - warmup_epochs 
    
    # 2. Dirichlet KL Penalty (False Evidence Suppression)
    # Warmup over the first 20% of Phase 2
    annealing_coef = min(1.0, active_epoch / (total_active_epochs * 0.2)) 
    alp_false = E * (1.0 - label) + 1.0
    L_KL = annealing_coef * dirichlet_kl_divergence(alp_false, c)
    
    # 3. Accuracy-Uncertainty Calibration (CUP)
    annealing_start = torch.tensor(0.01, dtype=torch.float32, device=logits.device)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_active_epochs * active_epoch)
    
    pred_scores, pred_cls = torch.max(expected_p, 1, keepdim=True)
    uncertainty = c / S
    target = mask_flat.view(-1, 1).float()
    acc_match = torch.eq(pred_cls, target).float()
    
    acc_uncertain = -torch.log(pred_scores * (1.0 - uncertainty) + eps)
    inacc_certain = -torch.log((1.0 - pred_scores) * uncertainty + eps)
        
    L_AU = annealing_AU * acc_match * acc_uncertain + (1.0 - annealing_AU) * (1.0 - acc_match) * inacc_certain
    
    # Return Unified Phase 2 Loss
    return torch.mean(L_ace) + torch.mean(L_KL) + torch.mean(L_AU)

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
                if self.training:
                    skip_features = F.dropout2d(skip_features, p=0.10)
                    
                decoder_features = torch.cat([decoder_features, skip_features], dim=1)
                decoder_features = block(decoder_features)
                
            return self.final_conv(decoder_features)

class HierarchicalEvidentialUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, latent_dims=[1, 1, 1, 1], channels_per_block=[24, 48, 96, 192, 192, 192, 192, 192]):
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

    def forward_train_step(self, image, mask_one_hot, mask_labels, current_epoch, beta=1.0, n_samples=3, warmup_epochs=50):
            # 1. Prior and Posterior Distributions
            prior_out = self.prior(image, use_mean=False)
            posterior_in = torch.cat([mask_one_hot, image], dim=1)
            posterior_out = self.posterior(posterior_in, use_mean=False)
            
            # 2. VAE Structural Penalty
            vae_kl_loss = self.calculate_hierarchical_kl(prior_out['distributions'], posterior_out['distributions'])
            
            total_evidential_loss = 0.0
            
            # 3. Evidential Reconstruction via Latent Sampling
            for _ in range(n_samples):
                z_sample = [dist.rsample() for dist in posterior_out['distributions']]
                prior_z_out = self.prior(image, z_q=z_sample)
                logits = self.f_comb(prior_z_out['encoder_features'], prior_z_out['decoder_features'])
                
                sample_loss = staged_hierarchical_evidential_loss(
                    logits=logits, 
                    mask=mask_labels, 
                    current_epoch=current_epoch, 
                    warmup_epochs=warmup_epochs,
                    c=self.num_classes
                )
                total_evidential_loss += sample_loss
                
            mean_evidential_loss = total_evidential_loss / n_samples
            
        # 4. Enforce Target Capacity to prevent Latent Collapse (Absolute Margin)
            target_kl = torch.tensor(5.0, device=image.device)
            
            # FIX: Changed from ReLU to Abs. Punishes the network if KL drops BELOW 5.0.
            penalized_kl = torch.abs(vae_kl_loss - target_kl)
            
            loss = mean_evidential_loss + (beta * penalized_kl)
            
            return loss, mean_evidential_loss, vae_kl_loss
            
 
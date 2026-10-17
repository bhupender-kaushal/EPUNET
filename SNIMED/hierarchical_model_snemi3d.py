import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions import Normal, Independent, kl


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
    B = loss_map.size(0)
    loss_flat = loss_map.view(B, -1)
    k_num = max(1, int(k_percent * loss_flat.size(1)))

    logits = torch.log(loss_flat + 1e-8)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    perturbed_logits = (logits + gumbel_noise) / temperature

    _, indices = torch.topk(perturbed_logits, k_num, dim=-1)
    mask = torch.zeros_like(loss_flat).scatter_(-1, indices, 1.0)

    return mask, k_num

# def unreduced_evidential_loss_multiclass(logits, mask, c=16, eps=1e-10, ignore_index=255, max_evidence=200.0):
#     B, C, H, W = logits.shape
#     evidence = torch.clamp(F.softplus(logits), max=max_evidence)
#     alpha = evidence + 1.0
#     alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, c)
#     mask_flat = mask.reshape(-1)

#     valid_mask = (mask_flat != ignore_index)
#     safe_mask_flat = torch.where(valid_mask, mask_flat, torch.zeros_like(mask_flat))

#     S = torch.sum(alpha_flat, dim=1, keepdim=True)
#     E = alpha_flat - 1.0
#     expected_p = alpha_flat / S
#     label = F.one_hot(safe_mask_flat, num_classes=c).float()
#     valid_col = valid_mask.float().unsqueeze(1)

#     L_ace_map = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha_flat)), dim=1, keepdim=True) * valid_col

#     # Absent-Class Dice Prevention
#     p_v = expected_p * valid_col
#     y_v = label * valid_col
#     smooth = 1e-5
#     intersection = torch.sum(p_v * y_v, dim=0)
#     cardinality = torch.sum(p_v, dim=0) + torch.sum(y_v, dim=0)
#     dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
#     class_present = (torch.sum(y_v, dim=0) > 0).float()
#     valid_classes = torch.sum(class_present)
    
#     L_e_dice = 1.0 - (torch.sum(dice_per_class * class_present) / valid_classes) if valid_classes > 0 else 1.0 - dice_per_class.mean()

#     # Dynamic KL Masking for Sparse Multiclass
#     alp_false = E * (1.0 - label) + 1.0
#     raw_KL_map = dirichlet_kl_divergence(alp_false, c)
    
#     # Scale down KL penalty for channels that don't exist in this patch
#     class_present_spatial = class_present.view(1, c).expand(B * H * W, c)
#     kl_weight_mask = torch.where(class_present_spatial > 0, 1.0, 0.01)
    
#     # Weight KL by channel presence
#     weighted_KL = torch.sum(raw_KL_map * kl_weight_mask, dim=1, keepdim=True)
#     L_KL_map = weighted_KL * valid_col

#     pred_scores, pred_cls = torch.max(expected_p, 1, keepdim=True)
#     uncertainty = c / S
#     target = safe_mask_flat.view(-1, 1).float()
#     acc_match = torch.eq(pred_cls, target).float()
#     acc_uncertain = -torch.log(pred_scores * (1.0 - uncertainty) + eps)
#     inacc_certain = -torch.log((1.0 - pred_scores) * uncertainty + eps)
#     L_AU_map = (acc_match * acc_uncertain + (1.0 - acc_match) * inacc_certain) * valid_col

#     return L_ace_map.view(B, H, W), L_e_dice, L_KL_map.view(B, H, W), L_AU_map.view(B, H, W), valid_mask.view(B, H, W)

def unreduced_evidential_loss_multiclass(logits, mask, c=16, eps=1e-10, ignore_index=255, max_evidence=200.0, class_weights=None):
    B, C, H, W = logits.shape
    evidence = torch.clamp(F.softplus(logits), max=max_evidence)
    alpha = evidence + 1.0
    alpha_flat = alpha.permute(0, 2, 3, 1).contiguous().view(-1, c)
    mask_flat = mask.reshape(-1)

    valid_mask = (mask_flat != ignore_index)
    safe_mask_flat = torch.where(valid_mask, mask_flat, torch.zeros_like(mask_flat))

    S = torch.sum(alpha_flat, dim=1, keepdim=True)
    E = alpha_flat - 1.0
    expected_p = alpha_flat / S
    label = F.one_hot(safe_mask_flat, num_classes=c).float()
    valid_col = valid_mask.float().unsqueeze(1)

    # --- NOVEL UPDATE: Extract Per-Channel Dirichlet Variance ---
    # Math: Var(p_c) = [alpha_c * (S - alpha_c)] / [S^2 * (S + 1)]
    variance_per_class = (alpha_flat * (S - alpha_flat)) / ((S ** 2) * (S + 1.0))
    variance_map = variance_per_class.view(B, H, W, c).permute(0, 3, 1, 2)
    # ------------------------------------------------------------
    L_ace_map_raw = label * (torch.digamma(S) - torch.digamma(alpha_flat))
    if class_weights is not None:
        # Broadcast weights [c] -> [1, c] to match [N, c] raw loss map
        L_ace_map_raw = L_ace_map_raw * class_weights.view(1, c)
        
    L_ace_map = torch.sum(L_ace_map_raw, dim=1, keepdim=True) * valid_col

    # L_ace_map = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha_flat)), dim=1, keepdim=True) * valid_col #updated for class weights

    # Absent-Class Dice Prevention
    p_v = expected_p * valid_col
    y_v = label * valid_col
    smooth = 1e-5
    intersection = torch.sum(p_v * y_v, dim=0)
    cardinality = torch.sum(p_v, dim=0) + torch.sum(y_v, dim=0)
    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
    class_present = (torch.sum(y_v, dim=0) > 0).float()
    valid_classes = torch.sum(class_present)
    
    L_e_dice = 1.0 - (torch.sum(dice_per_class * class_present) / valid_classes) if valid_classes > 0 else 1.0 - dice_per_class.mean()

    # Dynamic KL Masking for Sparse Multiclass
    alp_false = E * (1.0 - label) + 1.0
    raw_KL_map = dirichlet_kl_divergence(alp_false, c)
    
    class_present_spatial = class_present.view(1, c).expand(B * H * W, c)
    kl_weight_mask = torch.where(class_present_spatial > 0, 1.0, 0.01)
    
    weighted_KL = torch.sum(raw_KL_map * kl_weight_mask, dim=1, keepdim=True)
    L_KL_map = weighted_KL * valid_col

    pred_scores, pred_cls = torch.max(expected_p, 1, keepdim=True)
    uncertainty = c / S
    target = safe_mask_flat.view(-1, 1).float()
    acc_match = torch.eq(pred_cls, target).float()
    acc_uncertain = -torch.log(pred_scores * (1.0 - uncertainty) + eps)
    inacc_certain = -torch.log((1.0 - pred_scores) * uncertainty + eps)
    L_AU_map = (acc_match * acc_uncertain + (1.0 - acc_match) * inacc_certain) * valid_col

    # Return the variance_map alongside your standard losses
    return L_ace_map.view(B, H, W), L_e_dice, L_KL_map.view(B, H, W), L_AU_map.view(B, H, W), valid_mask.view(B, H, W), variance_map


def _safe_num_groups(channels, target=8):
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_channels=None):
        super(ResBlock, self).__init__()
        if down_channels is None: down_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, down_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_safe_num_groups(down_channels), down_channels)
        self.conv2 = nn.Conv2d(down_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_safe_num_groups(out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        def shield(tensor):
            return torch.nan_to_num(torch.clamp(tensor, min=-65000.0, max=65000.0), nan=0.0)

        residual = shield(self.shortcut(x))
        
        x = shield(self.conv1(x))
        x = self.relu(self.norm1(x))
        
        x = shield(self.conv2(x))
        x = self.norm2(x)
        
        return self.relu(shield(x + residual))


class StitchingDecoder(nn.Module):
    def __init__(self, latent_dims, channels_per_block, num_classes, num_resblocks=3):
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
            self.decoder_blocks.append(nn.Sequential(*[ResBlock(in_ch if j == 0 else out_ch, out_ch) for j in range(num_resblocks)]))
            curr_dec_channels = out_ch

        self.final_conv = nn.Conv2d(curr_dec_channels, num_classes, kernel_size=1)

    def forward(self, encoder_features, decoder_features):
        rev_encoder_outputs = encoder_features[::-1]
        start_level = len(rev_encoder_outputs) - len(self.decoder_blocks)

        for i, block in enumerate(self.decoder_blocks):
            level_idx = start_level + i
            decoder_features = F.interpolate(decoder_features, scale_factor=2, mode='bilinear', align_corners=False)
            skip_features = rev_encoder_outputs[level_idx]
            decoder_features = torch.cat([decoder_features, skip_features], dim=1)
            decoder_features = block(decoder_features)

        return self.final_conv(decoder_features)


class ScalarHierarchicalCore(nn.Module):
    def __init__(self, in_channels, latent_dims, channels_per_block, num_resblocks=3):
        super().__init__()
        self.latent_dims = latent_dims
        self.num_levels = len(channels_per_block)
        self.num_latent_levels = len(latent_dims)

        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        for out_channels in channels_per_block:
            self.encoder_blocks.append(
                nn.Sequential(*[ResBlock(curr_channels if j == 0 else out_channels, out_channels) for j in range(num_resblocks)])
            )
            curr_channels = out_channels

        self.dist_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        rev_channels = channels_per_block[::-1]
        curr_dec_channels = rev_channels[0]

        for i, latent_dim in enumerate(latent_dims):
            dist_layer = nn.Conv2d(curr_dec_channels, 2 * latent_dim, kernel_size=1)
            torch.nn.init.zeros_(dist_layer.weight)
            torch.nn.init.zeros_(dist_layer.bias)
            self.dist_convs.append(dist_layer)

            next_skip_channels = rev_channels[i + 1]
            in_ch = curr_dec_channels + latent_dim + next_skip_channels
            out_ch = rev_channels[i + 1]
            self.decoder_blocks.append(nn.Sequential(*[ResBlock(in_ch if j == 0 else out_ch, out_ch) for j in range(num_resblocks)]))
            curr_dec_channels = out_ch
    def forward(self, inputs=None, use_mean=False, z_q=None, encoder_features=None):
        # Bypasses the encoder if features are already provided (MC loop)
        if encoder_features is None:
            encoder_features = []
            x = inputs
            for i, block in enumerate(self.encoder_blocks):
                x = block(x)
                encoder_features.append(x)
                if i != self.num_levels - 1:
                    x = F.avg_pool2d(x, kernel_size=2, stride=2)

        decoder_features = encoder_features[-1]
        distributions, used_latents = [], []
        rev_encoder_outputs = encoder_features[::-1]

        for i in range(self.num_latent_levels):
            latent_dim = self.latent_dims[i]
            B, _, H, W = decoder_features.shape

            pooled = F.adaptive_avg_pool2d(decoder_features, 1)
            mu_logsigma = self.dist_convs[i](pooled).view(B, 2 * latent_dim)

            with autocast(enabled=False):
                mu = torch.clamp(mu_logsigma[:, :latent_dim], min=-50.0, max=50.0)
                logsigma = torch.clamp(mu_logsigma[:, latent_dim:], min=-4.0, max=2.0)
                scale = torch.exp(logsigma) + 1e-7
                mu = mu.float()
                scale = scale.float()
                dist = Independent(Normal(loc=mu, scale=scale), 1)
                distributions.append(dist)

            # Use provided Z (Posterior), use Mean (Validation), or Sample (Prior)
            z = z_q[i] if z_q is not None else (dist.mean if use_mean else dist.rsample())
            z = torch.clamp(z, min=-100.0, max=100.0)
            used_latents.append(z)

            z_spatial = z.view(B, latent_dim, 1, 1).expand(-1, -1, H, W)
            dec_out_lo = torch.cat([z_spatial, decoder_features], dim=1)
            dec_out_hi = F.interpolate(dec_out_lo, scale_factor=2, mode='bilinear', align_corners=False)

            skip_features = rev_encoder_outputs[i + 1]
            if self.training:
                skip_features = F.dropout2d(skip_features, p=0.15)

            decoder_features = torch.cat([dec_out_hi, skip_features], dim=1)
            decoder_features = self.decoder_blocks[i](decoder_features)

        return {
            'decoder_features': decoder_features, 
            'encoder_features': encoder_features,
            'distributions': distributions, 
            'used_latents': used_latents
        }

class HierarchicalEvidentialUNetSNEMI3D(nn.Module):
    def __init__(self, input_channels=1, num_classes=16,
                 latent_dims=(40, 25, 15, 5),
                 channels_per_block=(32, 64, 128, 256, 256, 256, 256, 256, 256),
                 num_resblocks=3):
        super().__init__()
        latent_dims, channels_per_block = list(latent_dims), list(channels_per_block)
        self.prior = ScalarHierarchicalCore(input_channels, latent_dims, channels_per_block, num_resblocks=num_resblocks)
        self.posterior = ScalarHierarchicalCore(input_channels + num_classes, latent_dims, channels_per_block, num_resblocks=num_resblocks)
        self.f_comb = StitchingDecoder(latent_dims, channels_per_block, num_classes, num_resblocks=num_resblocks)
        self.num_classes = num_classes

        weights = torch.ones(num_classes, dtype=torch.float32)
        weights[0] = 10.0  # 10x penalty for background/membrane
        self.register_buffer('class_weights', weights)

    def sample(self, image, use_mean=False):
        prior_out = self.prior(image, use_mean=use_mean)
        return self.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])

    def calculate_hierarchical_kl(self, prior_dists, posterior_dists, free_bits=2.5):
        kl_sum = 0.0
        for p_dist, q_dist in zip(prior_dists, posterior_dists):
            kl_sum = kl_sum + torch.mean(kl.kl_divergence(q_dist, p_dist))
        return torch.clamp(kl_sum, min=free_bits)


    def forward_train_step(self, image, multi_mask_labels, lambda_geco, kappa,  n_samples=4, current_epoch=0, ignore_index=255,  kl_anneal_epochs=20, free_bits=2.5, max_kl_weight=0.5, debug_step_id=None, total_epochs=None):

        gt_mask = multi_mask_labels[:, 0, :, :]
        gt_one_hot = F.one_hot(torch.where(gt_mask == ignore_index, 0, gt_mask), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 1. Base Prior & Posterior passes
        prior_out = self.prior(image)
        enc_features = prior_out['encoder_features']
        
        posterior_out = self.posterior(torch.cat([gt_one_hot, image], dim=1))
        vae_kl_loss = self.calculate_hierarchical_kl(prior_out['distributions'], posterior_out['distributions'], free_bits=free_bits)

        # 2. Stochastic MC Loop (Aleatoric Variance)
        total_masked_data_loss = 0.0
        for _ in range(n_samples):
            # Select random synthetic annotator
            ann_idx = torch.randint(0, multi_mask_labels.size(1), (1,)).item()
            curr_mask = multi_mask_labels[:, ann_idx, :, :]

            # Decode from Posterior using cached encoder features
            z_sample = [dist.rsample() for dist in posterior_out['distributions']]
            prior_dec = self.prior(encoder_features=enc_features, z_q=z_sample)['decoder_features']
            logits_stoch = self.f_comb(enc_features, prior_dec)

            with torch.cuda.amp.autocast(enabled=False):
                L_ace_map, L_e_dice_stoch, _, _, _, _ = unreduced_evidential_loss_multiclass(
                    logits_stoch.float(), curr_mask, c=self.num_classes, ignore_index=ignore_index, class_weights=self.class_weights
                )
                ohem_mask, k_num = stochastic_ohem_mask(L_ace_map, k_percent=0.04)
                B = L_ace_map.size(0)
                masked_ace = (L_ace_map.view(B, -1) * ohem_mask).sum() / (k_num * B)
                total_masked_data_loss = total_masked_data_loss + masked_ace + (0.2 * L_e_dice_stoch)

        mean_data_loss = total_masked_data_loss / n_samples

        # 3. Deterministic Global Anchor (Posterior Mean)
        z_mean = [dist.mean for dist in prior_out['distributions']]
        prior_dec_det = self.prior(encoder_features=enc_features, z_q=z_mean)['decoder_features']
        logits_det = self.f_comb(enc_features, prior_dec_det)

        with torch.cuda.amp.autocast(enabled=False):
            _, L_e_dice_det, L_KL_map, L_AU_map, _, variance_map = unreduced_evidential_loss_multiclass(
                logits_det.float(), gt_mask, c=self.num_classes, ignore_index=ignore_index, class_weights=self.class_weights
            )
            pen_mask, k_num_pen = stochastic_ohem_mask(L_KL_map + L_AU_map, k_percent=0.04)
            B = L_KL_map.size(0)
            masked_kl_edl = (L_KL_map.view(B, -1) * pen_mask).sum() / (k_num_pen * B)
            masked_au_edl = (L_AU_map.view(B, -1) * pen_mask).sum() / (k_num_pen * B)

            # Dynamic Warmups
            # kl_weight = min(max_kl_weight, current_epoch / max(kl_anneal_epochs, 1))
            # lambda_dice_det = 0.5 + min(1.0, current_epoch / 100.0)
            
            # --- NOVEL UPDATE: Trapezoidal Decay Schedule ---
            if current_epoch < kl_anneal_epochs:
                # Phase 1 (Epoch 0-40): Ramp up. Prevents false confidence during random initialization.
                kl_weight = max_kl_weight * (current_epoch / max(kl_anneal_epochs, 1))
            elif current_epoch < 150:
                # Phase 2 (Epoch 40-150): Hold. Forces network to learn structural ambiguity.
                kl_weight = max_kl_weight
            else:
                # Phase 3 (Epoch 150+): Decay. Removes penalty for 1-pixel bleed, allowing high-confidence boundaries.
                decay_epochs = 100.0
                decay_factor = max(0.01, 1.0 - ((current_epoch - 150) / decay_epochs))
                kl_weight = max_kl_weight * decay_factor

            lambda_dice_det = 0.5 + min(1.0, current_epoch / 100.0)

            constraint_loss = lambda_geco * (mean_data_loss - kappa)
            loss = constraint_loss + kl_weight * masked_kl_edl + masked_au_edl + vae_kl_loss + (lambda_dice_det * L_e_dice_det)

        return loss, mean_data_loss, (masked_kl_edl + masked_au_edl), vae_kl_loss, L_e_dice_det
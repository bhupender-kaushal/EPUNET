import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import ndimage
from skimage.segmentation import watershed

from load_snemi3d_data import SNEMI3DPatches, NUM_CLASSES, IGNORE_INDEX
from hierarchical_model_snemi3d import HierarchicalEvidentialUNetSNEMI3D
from utils import l2_regularisation

# ==========================================
# INFERENCE & CLUSTERING FUNCTIONS
# ==========================================
def greedy_hamming_clustering(masked_preds, threshold=8, ignore_index=255):

    S, H, W = masked_preds.shape
    flat_preds = masked_preds.view(S, -1).t()
    
    N_pixels = flat_preds.size(0)
    instance_map = torch.zeros(N_pixels, dtype=torch.long, device=masked_preds.device)
    
    valid_pixel_mask = (flat_preds != ignore_index).any(dim=1)
    unassigned_indices = torch.nonzero(valid_pixel_mask).squeeze(1)
    
    current_instance_id = 1
    
    while unassigned_indices.numel() > 0:
        seed_idx = unassigned_indices[0]
        seed_signature = flat_preds[seed_idx].unsqueeze(0) 
        
        unassigned_signatures = flat_preds[unassigned_indices] 
        valid_comparisons = (unassigned_signatures != ignore_index) & (seed_signature != ignore_index)
        mismatches = (unassigned_signatures != seed_signature) & valid_comparisons
        distances = mismatches.sum(dim=1) 
        
        cluster_mask = distances <= threshold
        cluster_indices = unassigned_indices[cluster_mask]
        
        instance_map[cluster_indices] = current_instance_id
        current_instance_id += 1
        
        unassigned_indices = unassigned_indices[~cluster_mask]
        
    return instance_map.view(H, W)

# ==========================================
# METRICS & UTILS
# ==========================================
def compute_ged(predictions, ground_truths, num_classes=16, ignore_index=255):
    B, S_p, H, W = predictions.shape
    S_g = ground_truths.shape[1]

    def _calc_distance_matrix(set_A, set_B, size_A, size_B):
        A_flat = set_A.view(B, size_A, -1)
        B_flat = set_B.view(B, size_B, -1)
        dist_matrix = torch.zeros(B, size_A, size_B, device=predictions.device)
        
        for c in range(1, num_classes): 
            A_c = (A_flat == c).float()
            B_c = (B_flat == c).float()
            
            valid_mask = (set_B.view(B, size_B, -1) != ignore_index).float()
            A_c = A_c * valid_mask
            
            for i in range(size_A):
                for j in range(size_B):
                    inter = torch.sum(A_c[:, i] * B_c[:, j], dim=1)
                    union = torch.sum(A_c[:, i], dim=1) + torch.sum(B_c[:, j], dim=1) - inter
                    iou = (inter + 1e-5) / (union + 1e-5)
                    dist_matrix[:, i, j] += (1.0 - iou)
                    
        return dist_matrix / (num_classes - 1)

    dist_pred_gt = _calc_distance_matrix(predictions, ground_truths, S_p, S_g)
    E_Y_Yhat = dist_pred_gt.mean(dim=(1, 2))

    dist_pred_pred = _calc_distance_matrix(predictions, predictions, S_p, S_p)
    E_Yhat_Yhat = dist_pred_pred.mean(dim=(1, 2))

    dist_gt_gt = _calc_distance_matrix(ground_truths, ground_truths, S_g, S_g)
    E_Y_Y = dist_gt_gt.mean(dim=(1, 2))

    ged_per_sample = 2.0 * E_Y_Yhat - E_Yhat_Yhat - E_Y_Y
    return torch.clamp(ged_per_sample, min=0.0).mean()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# INITIALIZATION
# ==========================================
seed = 42
set_seed(seed)

IMAGES_PATH = "/bhupender/datasets/SNEMI3D/image/train-input.tif"
LABELS_PATH = "/bhupender/datasets/SNEMI3D/seg/train-labels.tif"

train_dataset = SNEMI3DPatches(IMAGES_PATH, LABELS_PATH, mode="train")
val_dataset = SNEMI3DPatches(IMAGES_PATH, LABELS_PATH, mode="val")

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HierarchicalEvidentialUNetSNEMI3D(
    input_channels=1, 
    num_classes=NUM_CLASSES,
    latent_dims=(40, 25, 15, 5), # Restored for permutation capacity
    channels_per_block=(32, 64, 128, 256, 256, 256), 
    num_resblocks=2 
).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
epochs = 300
scaler = torch.cuda.amp.GradScaler()

timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = f"outputs_snemi3d/{seed}_checkpoint_{timestamp}"
os.makedirs(checkpoint_dir, exist_ok=True)
log_file_path = os.path.join(checkpoint_dir, "output.txt")

kappa, lambda_geco, lambda_lr, lambda_min, lambda_max = 3.5, 1.0, 0.001, 0.1, 30.0
KL_ANNEAL_EPOCHS = 40
best_val_ged = float("inf")
kappa_min = 0.2
kappa_momentum = 0.95

print("Starting SNEMI3D Hierarchical Evidential Training (Dynamic GECO + True GED Val)...")

for epoch in range(epochs):
    # ==========================================
    # TRAINING LOOP
    # ==========================================
    net.train()
    epoch_loss, epoch_data, epoch_edl_reg, epoch_vae_kl, epoch_det_dice = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, (patch, _, multi_mask_labels, meta) in enumerate(pbar):
        patch = patch.to(device)
        multi_mask_labels = multi_mask_labels.to(device).long()

        with torch.cuda.amp.autocast():
            final_loss, mean_data_loss, edl_reg_loss, vae_kl_loss, det_dice_loss = net.forward_train_step(
                image=patch,
                multi_mask_labels=multi_mask_labels,
                lambda_geco=lambda_geco,
                kappa=kappa,
                n_samples=4,
                current_epoch=epoch,
                total_epochs=epochs,
                ignore_index=IGNORE_INDEX,
                kl_anneal_epochs=KL_ANNEAL_EPOCHS,
                debug_step_id=f"epoch{epoch+1}_step{step}"
            )
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.f_comb)
            total_loss = final_loss + 1e-5 * reg_loss

        if not torch.isfinite(total_loss):
            print(f"[warn] epoch {epoch+1} step {step}: non-finite total_loss ({total_loss.item()}), skipping batch")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step_data_loss = mean_data_loss.item()
        with torch.no_grad():
            if np.isfinite(step_data_loss):
                constraint_violation = step_data_loss - kappa
                lambda_geco = min(max(lambda_min, lambda_geco * np.exp(lambda_lr * constraint_violation)), lambda_max)
                if step_data_loss < kappa:
                    target_kappa = step_data_loss * 0.95
                    kappa = max(kappa_min, kappa * kappa_momentum + target_kappa * (1.0 - kappa_momentum))

        valid_steps += 1
        epoch_loss += total_loss.item()
        epoch_data += mean_data_loss.item()
        epoch_edl_reg += edl_reg_loss.item()
        epoch_vae_kl += vae_kl_loss.item()
        epoch_det_dice += det_dice_loss.item()

        pbar.set_postfix({
            'Lam': f'{lambda_geco:.2f}',
            'Kap': f'{kappa:.2f}',
            'Data': f'{mean_data_loss.item():.2f}',
            'Dice': f'{det_dice_loss.item():.2f}',
            'VAE_KL': f'{vae_kl_loss.item():.2f}',
            'EDL_Reg': f'{edl_reg_loss.item():.2f}'
        })

    # ==========================================
    # VALIDATION LOOP (VARIANCE-MASKED GED)
    # ==========================================
    net.eval()
    val_data_sum, val_dice_sum, val_vae_kl_sum, val_edl_reg_sum, val_ged_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    val_batches = 0
    num_mc_samples = 4
    variance_threshold = 0.10 # Max allowable Dirichlet variance for valid prediction
    
    with torch.no_grad():
        for patch, _, multi_mask_labels, _ in val_loader:
            patch = patch.to(device)
            multi_mask_labels = multi_mask_labels.to(device).long()
            
            with torch.cuda.amp.autocast():
                _, v_data, v_edl_reg, v_vae_kl, v_dice = net.forward_train_step(
                    image=patch, 
                    multi_mask_labels=multi_mask_labels,
                    lambda_geco=lambda_geco, 
                    kappa=kappa, 
                    n_samples=num_mc_samples, 
                    current_epoch=epoch,
                    total_epochs=epochs,
                    ignore_index=IGNORE_INDEX,
                    kl_anneal_epochs=KL_ANNEAL_EPOCHS,
                    max_kl_weight=0.0 
                )
            
            sampled_preds = []
            prior_out = net.prior(patch)
            enc_cache = prior_out['encoder_features']
            
            with torch.cuda.amp.autocast():
                for _ in range(num_mc_samples):
                    z_sample = [dist.rsample() for dist in prior_out['distributions']]
                    prior_dec = net.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
                    logits = net.f_comb(enc_cache, prior_dec)
                    
                    # --- NOVEL UPDATE: Evidential Variance Masking ---
                    evidence = torch.clamp(F.softplus(logits), max=200.0)
                    alpha = evidence + 1.0
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    expected_p = alpha / S
                    
                    # Predict class based on expected probability
                    _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
                    
                    # Math: Extract variance for the specifically predicted class
                    alpha_c = torch.gather(alpha, 1, pred_cls)
                    var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
                    
                    # Mask pixels exceeding the variance threshold with IGNORE_INDEX (255)
                    valid_mask = (var_c < variance_threshold).float()
                    pred_cls_masked = torch.where(valid_mask == 1, pred_cls, torch.tensor(IGNORE_INDEX, device=device))
                    
                    sampled_preds.append(pred_cls_masked.squeeze(1))
                    # ------------------------------------------------
            
            # Shape: [B, 4, H, W]
            sampled_preds = torch.stack(sampled_preds, dim=1) 

            # compute_ged inherently skips IGNORE_INDEX during IoU calculation.
            # Thus, GED is now evaluated strictly on the confident core morphology, 
            # unaffected by boundary permutation noise.
            ged_score = compute_ged(sampled_preds, multi_mask_labels, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
            
            if torch.isfinite(v_data):
                val_data_sum += v_data.item()
                val_dice_sum += v_dice.item()
                val_vae_kl_sum += v_vae_kl.item()
                val_edl_reg_sum += v_edl_reg.item()
                val_ged_sum += ged_score.item()
                val_batches += 1

    val_data = val_data_sum / max(val_batches, 1)
    val_dice = val_dice_sum / max(val_batches, 1)
    val_vae = val_vae_kl_sum / max(val_batches, 1)
    val_edl = val_edl_reg_sum / max(val_batches, 1)
    val_ged = val_ged_sum / max(val_batches, 1)

    # ==========================================
    # CHECKPOINTING
    # ==========================================
    checkpoint_state = {
        "epoch": epoch + 1, 
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(), 
        "scaler": scaler.state_dict(), 
        "val_ged": val_ged,
        "val_data": val_data
    }
    torch.save(checkpoint_state, os.path.join(checkpoint_dir, "model_latest.pth"))

    if val_ged < best_val_ged:
        best_val_ged = val_ged
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, "model_best.pth"))
        log_msg = (
            f"[*] Epoch [{epoch+1}/{epochs}] | New Best! Val GED: {val_ged:.5f} | "
            f"Data: {val_data:.4f} | Dice: {val_dice:.4f} | "
            f"VAE_KL: {val_vae:.4f} | EDL_Reg: {val_edl:.4f}\n"
        )
    else:
        log_msg = (
            f"Epoch [{epoch+1}/{epochs}] | Val GED: {val_ged:.5f} | "
            f"Data: {val_data:.4f} | Dice: {val_dice:.4f} | "
            f"VAE_KL: {val_vae:.4f} | EDL_Reg: {val_edl:.4f}\n"
        )

    print(log_msg.strip())
    with open(log_file_path, "a") as f:
        f.write(log_msg)
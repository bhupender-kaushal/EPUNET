import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from load_LIDC_data import LIDC_IDRI_Folder
from hierarchical2 import HierarchicalEvidentialUNet
from utils import l2_regularisation 

def set_seed(seed=42):
    print(f"[*] Setting random seed to: {seed}")
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calc_ged_pytorch(samples, gts, smooth=1e-5):
    # Flatten spatial dimensions
    s_flat = samples.view(samples.size(0), -1) 
    y_flat = gts.view(gts.size(0), -1)         
    
    def get_pairwise_distances(A, B):
        # Efficient batch intersection and union
        inter = torch.mm(A, B.t())
        union = A.sum(dim=1, keepdim=True) + B.sum(dim=1, keepdim=True).t() - inter
        
        # FIX: Laplace smoothing mathematically replaces torch.where
        # Handles union=0 (empty patches) by returning iou=1.0 -> dist=0.0
        iou = (inter + smooth) / (union + smooth)
        dist = 1.0 - iou
        
        return dist
        
    d_sy = get_pairwise_distances(s_flat, y_flat)
    d_ss = get_pairwise_distances(s_flat, s_flat)
    d_yy = get_pairwise_distances(y_flat, y_flat)
    
    n, m = samples.size(0), gts.size(0)
    term1 = (2.0 / (n * m)) * d_sy.sum()
    term2 = (1.0 / (n ** 2)) * d_ss.sum()
    term3 = (1.0 / (m ** 2)) * d_yy.sum()
    
    return (term1 - term2 - term3).item()

# --- 1. Setup & Data ---
seed = 42
set_seed(seed)

# SET THIS TO YOUR TARGET DIRECTORY TO RESUME, OR None FOR A FRESH RUN
# Example: RESUME_DIR = "outputs/42_checkpoint_20260423_081530"
RESUME_DIR = 'outputs/42_checkpoint_20260422_131855'
# RESUME_DIR = 'outputs/42_checkpoint_20260422_131855'


train_path = 'data/train/'
val_path = 'data/val/' 

train_dataset = LIDC_IDRI_Folder(train_path, mode='train')
val_dataset = LIDC_IDRI_Folder(val_path, mode='test') 

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# --- 2. Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HierarchicalEvidentialUNet(input_channels=1, num_classes=2, latent_dims=[4,3,2,1], channels_per_block=[24, 48, 96, 192, 192, 192, 192, 192])
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-5) 
epochs = 500

# --- 3. GECO & Checkpoint Setup ---
kappa = 0.27         # TARGET BUMP: Set to 0.22 to force the phase transition
lambda_geco = 1.0    
lambda_lr = 0.01
lambda_min = 2.0
lambda_max = 50.0

best_val_ged = float('inf')
start_epoch = 0

if RESUME_DIR is not None and os.path.exists(RESUME_DIR):
    checkpoint_dir = RESUME_DIR
    latest_path = os.path.join(checkpoint_dir, "model_best.pth")
    # latest_path = os.path.join(checkpoint_dir, "good_model_31_65.pth")

    if os.path.exists(latest_path):
        print(f"[*] Resuming from checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_ged = checkpoint.get('val_ged', float('inf'))
        lambda_geco = checkpoint.get('lambda_geco', 50.0) # Restore Lam state
        print(f"[*] Resumed at Epoch {start_epoch} | Lam={lambda_geco:.3f}")
    log_file_path = os.path.join(checkpoint_dir, "output.txt")
else:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f"outputs/{seed}_checkpoint_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(checkpoint_dir, "output.txt")
    with open(log_file_path, "w") as f:
        f.write("Logging the model training process...\n") 
        f.write(f"Seed: {seed} | Warmup Epochs: 50 | Kappa: {kappa}\n")
        f.write("-" * 50 + "\n")

print(f"[*] Starting Hierarchical Evidential Training. Tracking Data Loss against Kappa = {kappa}")

# --- 4. Training Loop ---
for epoch in range(start_epoch, epochs):
    net.train()
    # Added epoch_data_loss to accumulators
    epoch_loss, epoch_data, epoch_edl, epoch_kl, epoch_det_dice = 0.0, 0.0, 0.0, 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, (patch, mask, _) in enumerate(pbar): 

        patch = patch.to(device)
        mask_labels = mask.to(device).long() 
        mask_one_hot = F.one_hot(mask_labels.squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
                
        final_loss, mean_data_loss, l_edl, l_kl, l_det_dice = net.forward_train_step(
            image=patch, 
            mask_one_hot=mask_one_hot, 
            mask_labels=mask_labels, 
            lambda_geco=lambda_geco,
            kappa=kappa,
            n_samples=4
        )
        
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.f_comb)
        total_loss = final_loss + 1e-5 * reg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()
        
        # --- GECO UPDATE RULE ---
        with torch.no_grad():
            constraint_violation = mean_data_loss.item() - kappa
            lambda_geco = lambda_geco * np.exp(lambda_lr * constraint_violation)
            lambda_geco = min(max(lambda_min, lambda_geco), lambda_max)
        
        steps = step + 1
        
        # Accumulate all metrics correctly
        epoch_loss += total_loss.item()
        epoch_data += mean_data_loss.item() # TRACK DATA
        epoch_edl += l_edl.item()
        epoch_kl += l_kl.item()
        epoch_det_dice += l_det_dice.item()
        
        # Print DATA to the terminal so you can watch the math happen
        pbar.set_postfix({
            'Loss': f'{epoch_loss/steps:.4f}', 
            'Data': f'{epoch_data/steps:.4f}',  # Watch this drop below kappa
            'EDL': f'{epoch_edl/steps:.4f}', 
            'KL': f'{epoch_kl/steps:.4f}',
            'DetDice': f'{epoch_det_dice/steps:.4f}',
            'Lam': f'{lambda_geco:.3f}'  
        })
        
    # --- 5. STOCHASTIC GED VALIDATION ---
    net.eval()
    total_ged_sum = 0.0
    total_images = 0
    N_val_samples = 8
    
    with torch.no_grad():
        for patch, masks, _ in val_loader:
            patch = patch.to(device)
            masks = masks.to(device).float() 
            B = patch.size(0)
            
            batch_samples = []
            for _ in range(N_val_samples):
                logits = net.sample(patch, use_mean=False) 
                
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                
                pred_fg_prob = probs[:, 1:2, :, :] 
                pred_binary = (pred_fg_prob > 0.505).float() 
                batch_samples.append(pred_binary)
                
            for i in range(B):
                samples_for_image = torch.cat([s[i] for s in batch_samples], dim=0)
                gts_for_image = masks[i] 
                total_ged_sum += calc_ged_pytorch(samples_for_image, gts_for_image)
            total_images += B
            
    val_ged = total_ged_sum / total_images
    
    # --- 6. CHECKPOINT SAVER & SAFE LOGGING ---
    checkpoint_state = {
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lambda_geco': lambda_geco,          # Ensure Lambda is saved!
        'val_ged': val_ged
    }

    latest_path = os.path.join(checkpoint_dir, "model_latest.pth")
    torch.save(checkpoint_state, latest_path)
    
    if val_ged < best_val_ged:
        best_val_ged = val_ged
        best_path = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(checkpoint_state, best_path)
        log_msg = f"[*] Epoch [{epoch+1}/{epochs}] | New Best! Avg Val GED: {val_ged:.5f}. Saved to model_best.pth\n"
    else:
        log_msg = f"Epoch [{epoch+1}/{epochs}] | Avg Val GED: {val_ged:.5f}\n"
        
    print(log_msg.strip())
    
    with open(log_file_path, "a") as f:
        f.write(log_msg)
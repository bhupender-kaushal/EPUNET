import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import glob
from torch.utils.data import DataLoader

# Replace with your actual imports
from load_LIDC_data import LIDC_IDRI_Folder
from probabilistic_unet import ProbabilisticUnet

output_base_dir = "test_results_ood"
os.makedirs(output_base_dir, exist_ok=True)

def iou_distance(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return 1.0 - (intersection / union)

def calculate_ged(sample_list, gt_list):
    n, m = len(sample_list), len(gt_list)
    term1 = (2.0 / (n * m)) * sum(iou_distance(s, y) for s in sample_list for y in gt_list)
    term2 = (1.0 / (n**2)) * sum(iou_distance(s1, s2) for s1 in sample_list for s2 in sample_list)
    term3 = (1.0 / (m**2)) * sum(iou_distance(y1, y2) for y1 in gt_list for y2 in gt_list)
    return term1 - term2 - term3

def analyze_single_patient_ood(net, test_loader, device, N_samples=8, threshold=0.5):
    net.eval()
    
    # 1. Grab exactly ONE patient from the dataset
    data_iter = iter(test_loader)
    patch_orig, masks_orig, meta = next(data_iter)
    patch_orig = patch_orig.to(device)
    masks = masks_orig.to(device)
    
    # Store the true min and max to lock the visual brightness later
    v_min, v_max = patch_orig.min().item(), patch_orig.max().item()

    # 2. Create our 4 exact variations of this single patient
    perturbations = []
    
    # A. Normal
    perturbations.append(("Normal", patch_orig.clone()))
    
    # B. Rotated 90 Degrees
    perturbations.append(("Rotated 90°", torch.rot90(patch_orig.clone(), k=1, dims=[2, 3])))
    
    # C. Severe Gaussian Noise (scaled to the image's standard deviation)
    noise = torch.randn_like(patch_orig) * patch_orig.std() * 1.0
    noisy_patch = patch_orig.clone() + noise
    perturbations.append(("Noisy Sensor", noisy_patch))
    
    # D. Local Cutout (Using the image's minimum value so it looks like outside air)
    cutout_patch = patch_orig.clone()
    c = cutout_patch.shape[-1] // 2
    cutout_patch[:, :, c-20:c+20, c-20:c+20] = v_min
    perturbations.append(("Missing Data (Cutout)", cutout_patch))

    all_results = []

    # 3. Run the network on each variation
    for name, patch in perturbations:
        with torch.no_grad():
            net.forward(patch, None, training=False)
            
            # Latent KL Score Calculation
            prior_dist = net.prior_latent_space
            dummy_mu = torch.zeros_like(prior_dist.base_dist.mean)
            dummy_sigma = torch.ones_like(prior_dist.base_dist.scale)
            standard_normal = Independent(Normal(loc=dummy_mu, scale=dummy_sigma), 1)
            latent_ood_score = kl.kl_divergence(prior_dist, standard_normal).mean().item()
            
            sample_masks, prob_outputs, epistemic_outputs = [], [], []
            
            for _ in range(N_samples):
                logits = net.sample(testing=True, temperature=0.8) 
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                
                prob = (alpha / S)[0, 1].cpu().numpy()
                prob_outputs.append(prob)
                sample_masks.append((prob > threshold).astype(np.float32))
                
                # Epistemic Uncertainty
                u = (2.0 / S)[0, 0].cpu().numpy()
                epistemic_outputs.append(u)
                
            prob_stack = np.stack(prob_outputs)
            reconstruction = (np.mean(prob_stack, axis=0) > threshold).astype(float)
            expected_epistemic = np.mean(np.stack(epistemic_outputs), axis=0)

            all_results.append({
                "name": name,
                "patch": patch[0, 0].cpu().numpy(),
                "recon": reconstruction,
                "samples": sample_masks[:4],
                "epistemic": expected_epistemic,
                "latent_score": latent_ood_score
            })

    # --- PLOTTING ---
    num_cols = 4
    num_rows = 1 + 1 + 4 + 1 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.8))
    fig.suptitle("Evidential OOD Analysis (Single Patient Comparison)", fontsize=20, fontweight='bold', y=1.02)

    labels = ["Input", "Mean Recon", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Epistemic Uncertainty"]

    for col in range(num_cols):
        res = all_results[col]
        
        axes[0, col].set_title(f"{res['name']}\nLatent OOD Score: {res['latent_score']:.2f}", 
                               fontsize=14, pad=10, fontweight='bold')
        
        # LOCKED INTENSITY: Using vmin and vmax so brightness never shifts
        axes[0, col].imshow(res["patch"], cmap='gray', vmin=v_min, vmax=v_max)
        axes[1, col].imshow(res["recon"], cmap='gray')
        
        for i in range(4): 
            axes[2+i, col].imshow(res["samples"][i], cmap='gray')
            
        # FIXED COLORMAP: Magma creates a beautiful black->orange glow with zero green
        axes[6, col].imshow(res["epistemic"], cmap='magma', vmin=0, vmax=1.0)
        
        for row in range(num_rows):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(labels[row], fontweight='bold', rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, "single_patient_ood_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

def load_latest_checkpoint(net, checkpoint_dir, device="cpu"):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.pt*"))
    if not list_of_files: raise FileNotFoundError("No checkpoints found.")
    latest_file = max(list_of_files, key=os.path.getctime)
    state_dict = torch.load(latest_file, map_location=device)
    
    if 'model_state_dict' in state_dict: net.load_state_dict(state_dict['model_state_dict'])
    elif 'state_dict' in state_dict: net.load_state_dict(state_dict['state_dict'])
    else: net.load_state_dict(state_dict)
    
    return net

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Update to your proposed model checkpoint directory
    checkpoint_dir = "/bhupender/new_punet/checkpoints_20260302_143923" 

    test_path  = 'data/test/'
    test_dataset = LIDC_IDRI_Folder(test_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Matching the proposed architecture
    net = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[32,64,128,192], 
                            latent_dim=6, no_convs_fcomb=4, beta=0.1)
    net.to(device)
    net = load_latest_checkpoint(net, checkpoint_dir, device=device)
    
    print("[*] Running Dual Safety OOD Analysis...")
    analyze_single_patient_ood(net, test_loader, device, N_samples=8,)
import os
import torch
import numpy as np
import matplotlib.subplots as plt
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import glob
from torch.utils.data import DataLoader

from load_LIDC_data import LIDC_IDRI_Folder
from hierarchical_model import HierarchicalEvidentialUNet

output_base_dir = "test_results_ood"
os.makedirs(output_base_dir, exist_ok=True)

def analyze_single_patient_ood(net, test_loader, device, N_samples=8, threshold=0.5):
    net.eval()
    
    data_iter = iter(test_loader)
    patch_orig, masks_orig, meta = next(data_iter)
    patch_orig = patch_orig.to(device)
    
    v_min, v_max = patch_orig.min().item(), patch_orig.max().item()

    perturbations = []
    perturbations.append(("Normal", patch_orig.clone()))
    perturbations.append(("Rotated 90°", torch.rot90(patch_orig.clone(), k=1, dims=[2, 3])))
    
    noise = torch.randn_like(patch_orig) * patch_orig.std() * 1.0
    perturbations.append(("Noisy Sensor", patch_orig.clone() + noise))
    
    cutout_patch = patch_orig.clone()
    c = cutout_patch.shape[-1] // 2
    cutout_patch[:, :, c-20:c+20, c-20:c+20] = v_min
    perturbations.append(("Missing Data (Cutout)", cutout_patch))

    all_results = []

    for name, patch in perturbations:
        with torch.no_grad():
            # Get multi-scale distributions
            prior_out = net.prior(patch, use_mean=True)
            prior_dists = prior_out['distributions']
            
            # Aggregate Latent OOD Score across all scales
            latent_ood_score = 0.0
            for dist in prior_dists:
                dummy_mu = torch.zeros_like(dist.base_dist.loc)
                dummy_sigma = torch.ones_like(dist.base_dist.scale)
                standard_normal = Independent(Normal(loc=dummy_mu, scale=dummy_sigma), 1)
                latent_ood_score += kl.kl_divergence(dist, standard_normal).mean().item()
            
            sample_masks, prob_outputs, epistemic_outputs = [], [], []
            
            for _ in range(N_samples):
                logits = net.sample(patch, use_mean=False) 
                evidence = F.softplus(logits)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1, keepdim=True)
                
                prob = (alpha / S)[0, 1].cpu().numpy()
                prob_outputs.append(prob)
                sample_masks.append((prob > threshold).astype(np.float32))
                
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
    num_rows = 8 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.8))
    fig.suptitle("Evidential OOD Analysis", fontsize=20, fontweight='bold', y=1.02)

    labels = ["Input", "Mean Recon", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Epistemic Uncertainty"]

    for col in range(num_cols):
        res = all_results[col]
        axes[0, col].set_title(f"{res['name']}\nLatent OOD Score: {res['latent_score']:.2f}", fontsize=14, pad=10, fontweight='bold')
        axes[0, col].imshow(res["patch"], cmap='gray', vmin=v_min, vmax=v_max)
        axes[1, col].imshow(res["recon"], cmap='gray')
        
        for i in range(4): 
            axes[2+i, col].imshow(res["samples"][i], cmap='gray')
            
        axes[6, col].imshow(res["epistemic"], cmap='magma', vmin=0, vmax=1.0)
        
        for row in range(num_rows-1): # -1 to match label length
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(labels[row], fontweight='bold', rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, "single_patient_ood_analysis.png"), dpi=300, bbox_inches='tight')
    print("Plot saved.")

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
    checkpoint_dir = "checkpoints_xxx" # Update this
    test_path  = 'data/test/'
    test_dataset = LIDC_IDRI_Folder(test_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    net = HierarchicalEvidentialUNet(input_channels=1, num_classes=2, latent_dims=[1, 1, 1, 1], channels_per_block=[24, 48, 96, 192, 192, 192, 192, 192])
    net.to(device)
    net = load_latest_checkpoint(net, checkpoint_dir, device=device)
    
    analyze_single_patient_ood(net, test_loader, device, N_samples=8)
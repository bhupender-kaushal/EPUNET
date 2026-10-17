import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import glob
from torch.utils.data import DataLoader
import random
from scipy.optimize import linear_sum_assignment

# Replace with your actual imports
from load_LIDC_data import LIDC_IDRI_Folder
from hierarchical2 import HierarchicalEvidentialUNet

# --- UTILITY & METRIC MATH ---

def to_rgb(arr, cmap={0: (0, 0, 0), 1: (255, 255, 255)}):
    new_arr = np.zeros(shape=(arr.shape)+(3, ))
    for c in cmap.keys():
        ixs = np.where(arr == c)
        new_arr[ixs] = [cmap[c][i] / 255. for i in range(3)]
    return new_arr

def iou_distance(mask1, mask2, smooth=1e-5):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 1.0 - ((intersection + smooth) / (union + smooth))

def calculate_ged(sample_list, gt_list):
    n, m = len(sample_list), len(gt_list)
    if n == 0 or m == 0: return 0.0
    term1 = (2.0 / (n * m)) * sum(iou_distance(s, y) for s in sample_list for y in gt_list)
    term2 = (1.0 / (n**2)) * sum(iou_distance(s1, s2) for s1 in sample_list for s2 in sample_list)
    term3 = (1.0 / (m**2)) * sum(iou_distance(y1, y2) for y1 in gt_list for y2 in gt_list)
    return term1 - term2 - term3

def dice_coefficient(mask1, mask2, smooth=1e-5):
    intersection = np.logical_and(mask1, mask2).sum()
    total_area = mask1.sum() + mask2.sum()
    return (2.0 * intersection + smooth) / (total_area + smooth)

def kohl_iou(pred, gt):
    """Calculates IoU enforcing the Kohl rule for empty foregrounds."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    sum_pred = np.sum(pred_b)
    sum_gt = np.sum(gt_b)
    
    # Kohl's Empty Mask Rule
    if sum_gt == 0 and sum_pred == 0:
        return 1.0
    if sum_gt == 0 or sum_pred == 0:
        return 0.0
        
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return intersection / union

def hungarian_matched_iou(samples, gts):
    """Uses the Hungarian Algorithm to find optimal sample-to-GT assignments."""
    M = len(gts)
    N = len(samples)
    if M == 0 or N == 0: return 0.0
        
    # Cost matrix: M x N. 
    # Negate the IoU because linear_sum_assignment minimizes the cost.
    cost_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            cost_matrix[i, j] = -kohl_iou(samples[j], gts[i])
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_ious = [-cost_matrix[row, col] for row, col in zip(row_ind, col_ind)]
    return np.mean(matched_ious)

def get_patient_slice_id(meta, fallback_id=0):
    try:
        meta_str = str(meta[0]) if isinstance(meta, (list, tuple)) else str(meta)
        if "/" in meta_str or "\\" in meta_str or meta_str.endswith(".png") or meta_str.endswith(".npy"):
            meta_str = os.path.normpath(meta_str)
            patient_id = os.path.basename(os.path.dirname(meta_str))
            slice_name = os.path.basename(meta_str).split('.')[0] 
            return f"{patient_id}_Slice_{slice_name}"
        else:
            return f"{meta_str}_Slice_{fallback_id}"
    except Exception:
        pass
    return f"Patient_Slice_{fallback_id}"

# --- GLOBAL EVALUATION & BEST SLICE TRACKING ---

def evaluate_full_dataset(net, test_loader, device, output_dir, N_samples=16, threshold=0.505):
    net.eval()
    
    # Trackers for the two distinct evaluation constraints
    metrics_all = {'ged': [], 'dice': [], 'iou_rec': [], 'h_iou': []}
    metrics_4gt = {'ged': [], 'dice': [], 'iou_rec': [], 'h_iou': []}
    
    detailed_stats_path = os.path.join(output_dir, "per_patient_detailed_stats.txt")
    summary_stats_path = os.path.join(output_dir, "global_summary.txt")
    top_k_slices = []
    
    print(f"[*] Starting Full Dataset Evaluation. Logging detailed stats to {detailed_stats_path}")
    
    with open(detailed_stats_path, "w") as f:
        f.write("=== HIERARCHICAL EVIDENTIAL U-NET: PER-PATIENT STATS ===\n")
        f.write(f"{'Patient_Slice_ID':<40} | {'GED':<8} | {'DICE':<8} | {'IoU_rec':<8} | {'H_IoU':<8} | {'Valid_GTs':<10}\n")
        f.write("-" * 95 + "\n")
    
    with torch.no_grad():
        for batch_idx, (patch_orig, masks_orig, meta) in enumerate(test_loader):
            if masks_orig.shape[1] != 4: continue
                
            patch = patch_orig.to(device)
            # gt_list contains all 4 masks, including empty ones (valid expert gradings)
            gt_list = [masks_orig[0, j].cpu().numpy() for j in range(4)]
            patient_id = get_patient_slice_id(meta, fallback_id=batch_idx+1)
            
            # Count how many annotators physically drew a tumor
            num_tumor_gts = sum(1 for gt in gt_list if gt.sum() > 0)
            
            # --- 1. POSTERIOR RECONSTRUCTION (IoU_rec) ---
            posterior_recons = []
            slice_iou_rec_list = []
            
            for i in range(4):
                gt_tensor = masks_orig[:, i:i+1, :, :].to(device).float()
                
                # Force network through Posterior Q(z|X,Y) using mean
                logits_rec = net.reconstruct_from_posterior(patch, gt_tensor)
                
                # Evidential smoothing applied to posterior
                alpha_rec = F.softplus(logits_rec) + 1.0
                S_rec = torch.sum(alpha_rec, dim=1, keepdim=True)
                prob_rec = (alpha_rec / S_rec)[0, 1].cpu().numpy()
                pred_binary_rec = (prob_rec > threshold).astype(np.float32)
                
                posterior_recons.append(pred_binary_rec)
                
                # Calculate metric using Kohl's rule
                slice_iou_rec_list.append(kohl_iou(pred_binary_rec, gt_list[i]))
                
            avg_slice_iou_rec = np.mean(slice_iou_rec_list)

            # --- 2. PRIOR PREDICTION (Monte Carlo Integration) ---
            sample_probs = []
            sample_epistemics = []
            
            for _ in range(N_samples):
                logits_stoch = net.sample(patch, use_mean=False) 
                alpha_stoch = F.softplus(logits_stoch) + 1.0
                S_stoch = torch.sum(alpha_stoch, dim=1, keepdim=True)
                
                prob = (alpha_stoch / S_stoch)[0, 1].cpu().numpy()
                epistemic = (2.0 / S_stoch)[0, 0].cpu().numpy()
                
                sample_probs.append(prob)
                sample_epistemics.append(epistemic)
            
            mean_prob_map = np.mean(sample_probs, axis=0)
            mean_recon = (mean_prob_map > threshold).astype(float)
            expected_epistemic = np.mean(sample_epistemics, axis=0)
            
            sample_masks = [(p > threshold).astype(np.float32) for p in sample_probs]
            
            # --- 3. METRIC CALCULATIONS ---
            ged_score = calculate_ged(sample_masks, gt_list)
            h_iou = hungarian_matched_iou(sample_masks, gt_list)
            
            if num_tumor_gts > 0:
                sum_gt = np.sum(gt_list, axis=0)
                consensus_gt = (sum_gt >= (len(gt_list) / 2.0)).astype(float)
                dice_score = dice_coefficient(mean_recon, consensus_gt)
            else:
                # If no annotator found a tumor, and the model predicted empty, DICE is mathematically 1.0
                dice_score = 1.0 if mean_recon.sum() == 0 else 0.0
            
            # --- 4. RECORDING DATA ---
            # Append to Full Dataset metrics
            metrics_all['ged'].append(ged_score)
            metrics_all['dice'].append(dice_score)
            metrics_all['iou_rec'].append(avg_slice_iou_rec)
            metrics_all['h_iou'].append(h_iou)
            
            # Append to Strict 4-Annotator metrics
            if num_tumor_gts == 4:
                metrics_4gt['ged'].append(ged_score)
                metrics_4gt['dice'].append(dice_score)
                metrics_4gt['iou_rec'].append(avg_slice_iou_rec)
                metrics_4gt['h_iou'].append(h_iou)
            
            slice_data = {
                "id": patient_id, "patch": patch[0, 0].cpu().numpy(), "gt_list": gt_list,
                "posterior_recons": posterior_recons,
                "mean_recon": mean_recon, "samples": sample_masks[:16],
                "epistemic": expected_epistemic, "ged": ged_score, "dice": dice_score,
                "iou_rec": avg_slice_iou_rec, "h_iou": h_iou
            }
            
            # Only visualize cases where actual tumors exist for the top 5
            if num_tumor_gts > 0:
                top_k_slices.append(slice_data)
                top_k_slices = sorted(top_k_slices, key=lambda x: x['ged'])[:5]
            
            with open(detailed_stats_path, "a") as f:
                f.write(f"{patient_id:<40} | {ged_score:<8.4f} | {dice_score:<8.4f} | {avg_slice_iou_rec:<8.4f} | {h_iou:<8.4f} | {num_tumor_gts:<10}\n")
            
            if len(metrics_all['ged']) % 100 == 0: 
                print(f"   Evaluated {len(metrics_all['ged'])} total slices...")

    # --- 5. COMPILING FINAL STATISTICS ---
    with open(summary_stats_path, "w") as f:
        f.write("=== GLOBAL TEST SET PERFORMANCE ===\n")
        f.write(f"Total Slices Evaluated: {len(metrics_all['ged'])}\n")
        f.write(f"Avg GED:             {np.mean(metrics_all['ged']):.4f}\n")
        f.write(f"Avg Consensus DICE:  {np.mean(metrics_all['dice']):.4f}\n")
        f.write(f"Avg IoU_rec:         {np.mean(metrics_all['iou_rec']):.4f}\n")
        f.write(f"Avg Hungarian IoU:   {np.mean(metrics_all['h_iou']):.4f}\n\n")
        
        f.write("=== STRICT 4-ANNOTATOR PERFORMANCE ===\n")
        f.write(f"Total Slices Evaluated: {len(metrics_4gt['ged'])}\n")
        f.write(f"Avg GED:             {np.mean(metrics_4gt['ged']):.4f}\n")
        f.write(f"Avg Consensus DICE:  {np.mean(metrics_4gt['dice']):.4f}\n")
        f.write(f"Avg IoU_rec:         {np.mean(metrics_4gt['iou_rec']):.4f}\n")
        f.write(f"Avg Hungarian IoU:   {np.mean(metrics_4gt['h_iou']):.4f}\n")
        
    print(f"\n[*] Full evaluation complete. Summary saved to {summary_stats_path}")
    return np.mean(metrics_all['ged']), np.mean(metrics_all['dice']), np.mean(metrics_all['iou_rec']), top_k_slices

def plot_top_5_slices(top_slices, output_dir):
    if not top_slices: return
    num_cols = len(top_slices)
    num_rows = 27 
    print(f"[*] Generating Top {num_cols} performing slices visualization grid...")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.5))
    if num_cols == 1: axes = np.expand_dims(axes, axis=1) 
    fig.suptitle("Top 5 Performing Slices (Prior Prediction vs. Posterior Capacity)", fontsize=22, fontweight='bold', y=1.01)
    
    row_labels = (
        ["Input", "Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4"] + 
        ["Post. Recon 1", "Post. Recon 2", "Post. Recon 3", "Post. Recon 4"] + 
        ["Prior MC Recon"] + 
        [f"Prior Sample {i+1}" for i in range(16)] + 
        ["Epistemic Uncert."]
    )

    for col, data in enumerate(top_slices):
        id_str = str(data["id"]).replace("_Slice_", "\nSlice: ")
        axes[0, col].set_title(f"{id_str}\nGED: {data['ged']:.4f} | DICE: {data['dice']:.4f}\nIoU_rec: {data['iou_rec']:.4f} | H_IoU: {data['h_iou']:.4f}", fontsize=11, fontweight='bold', pad=10)
        
        axes[0, col].imshow(data['patch'], cmap='gray')
        
        for i in range(4): 
            axes[1+i, col].imshow(data['gt_list'][i], cmap='magma', vmin=0, vmax=1.0)
            
        for i in range(4): 
            axes[5+i, col].imshow(data['posterior_recons'][i], cmap='gray')
            
        axes[9, col].imshow(data['mean_recon'], cmap='gray')
        
        for i in range(16):
            if i < len(data['samples']): axes[10+i, col].imshow(data['samples'][i], cmap='gray')
            else: axes[10+i, col].imshow(np.zeros_like(data['patch']), cmap='gray')
            
        axes[26, col].imshow(data['epistemic'], cmap='magma', vmin=0, vmax=1.0)
        
        for row in range(num_rows):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0: axes[row, col].set_ylabel(row_labels[row], fontsize=12, fontweight='bold', rotation=0, labelpad=80, ha='right', va='center')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "top_5_performing_slices_grid.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- VISUALIZATION FUNCTIONS ---

def analyze_multiple_patients_performance(net, test_loader, device, output_dir, num_patients=5, N_samples=8, threshold=0.505):
    net.eval()
    data_iter = iter(test_loader)
    all_results = []
    while len(all_results) < num_patients:
        try: patch_orig, masks_orig, meta = next(data_iter)
        except StopIteration: break
        if masks_orig.shape[1] != 4 or sum(1 for j in range(4) if masks_orig[0, j].sum() > 0) == 0: continue
            
        patch = patch_orig.to(device)
        gt_list = [masks_orig[0, i].cpu().numpy() for i in range(4)]
        
        valid_gts = [gt for gt in gt_list if gt.sum() > 0]
        gt_combined = (np.sum(valid_gts, axis=0) >= (len(valid_gts)/2.0)).astype(float) if valid_gts else np.zeros_like(gt_list[0])
        patient_id = get_patient_slice_id(meta, fallback_id=len(all_results)+1)

        with torch.no_grad():
            sample_probs = []
            for _ in range(N_samples):
                logits_stoch = net.sample(patch, use_mean=False) 
                alpha_stoch = F.softplus(logits_stoch) + 1.0
                S_stoch = torch.sum(alpha_stoch, dim=1, keepdim=True)
                sample_probs.append((alpha_stoch / S_stoch)[0, 1].cpu().numpy())
            
            mean_recon = (np.mean(sample_probs, axis=0) > threshold).astype(float)
            sample_masks = [(p > threshold).astype(np.float32) for p in sample_probs]
            
            ged_score = calculate_ged(sample_masks, gt_list)
            dice_score = dice_coefficient(mean_recon, gt_combined) if valid_gts else 0.0

            all_results.append({
                "id": patient_id, "patch": patch[0, 0].cpu().numpy(), "gt_combined": gt_combined,
                "mean_recon": mean_recon, "samples": sample_masks[:4], "ged": ged_score, "dice": dice_score
            })

    if not all_results: return

    num_cols, num_rows = 7, len(all_results)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.0, num_rows * 3.0))
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Model Performance & Sample Variance Across Patients", fontsize=20, fontweight='bold', y=1.02)
    col_headers = ["Input Image", "Consensus GT", "MC Recon", "Sample 1", "Sample 2", "Sample 3", "Sample 4"]

    for row in range(num_rows):
        res = all_results[row]
        if row == 0:
            for col in range(num_cols): axes[row, col].set_title(col_headers[col], fontsize=14, fontweight='bold')
        id_str = str(res["id"]).replace("_Slice_", "\nSlice: ")
        axes[row, 0].set_ylabel(f"{id_str}\nGED: {res['ged']:.3f}\nDICE: {res['dice']:.3f}", fontsize=10, fontweight='bold', rotation=0, labelpad=60, ha='right', va='center')
        axes[row, 0].imshow(res["patch"], cmap='gray')
        axes[row, 1].imshow(res["gt_combined"], cmap='magma', vmin=0, vmax=1.0) 
        axes[row, 2].imshow(res["mean_recon"], cmap='gray')
        for i in range(4): axes[row, 3+i].imshow(res["samples"][i], cmap='gray')
        for col in range(num_cols): axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "multi_patient_variance_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_single_patient_ood(net, test_loader, device, output_dir, N_samples=8, threshold=0.505):
    net.eval()
    data_iter = iter(test_loader)
    while True:
        try:
            patch_orig, masks_orig, meta = next(data_iter)
            if masks_orig.shape[1] == 4 and not any(masks_orig[0, i].sum() == 0 for i in range(4)): break
        except StopIteration: return
            
    patch_orig = patch_orig.to(device)
    v_min, v_max = patch_orig.min().item(), patch_orig.max().item()

    patch_std = max(patch_orig.std().item(), 0.1) 
    
    perturbations = [
        ("Normal", patch_orig.clone()),
        ("Rotated 90 Deg", torch.rot90(patch_orig.clone(), k=1, dims=[2, 3])),
        ("Noisy Sensor", patch_orig.clone() + (torch.randn_like(patch_orig) * patch_std * 1.0))
    ]
    
    cutout_patch = patch_orig.clone()
    c = cutout_patch.shape[-1] // 2
    cutout_patch[:, :, c-20:c+20, c-20:c+20] = v_min
    perturbations.append(("Missing Data", cutout_patch))

    all_results = []
    for name, patch in perturbations:
        with torch.no_grad():
            sample_probs = []
            sample_epistemics = []
            for _ in range(N_samples):
                logits_stoch = net.sample(patch, use_mean=False) 
                alpha_stoch = F.softplus(logits_stoch) + 1.0
                S_stoch = torch.sum(alpha_stoch, dim=1, keepdim=True)
                sample_probs.append((alpha_stoch / S_stoch)[0, 1].cpu().numpy())
                sample_epistemics.append((2.0 / S_stoch)[0, 0].cpu().numpy())

            reconstruction = (np.mean(sample_probs, axis=0) > threshold).astype(float)
            expected_epistemic = np.mean(sample_epistemics, axis=0)
            sample_masks = [(p > threshold).astype(np.float32) for p in sample_probs]

            all_results.append({
                "name": name, "patch": patch[0, 0].cpu().numpy(), "recon": reconstruction,
                "samples": sample_masks[:4], "epistemic": expected_epistemic
            })

    num_cols, num_rows = 4, 7 
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 2.8))
    fig.suptitle("Evidential OOD Analysis (MC Epistemic Evaluation)", fontsize=20, fontweight='bold', y=1.02)
    labels = ["Input", "MC Recon", "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Epistemic Uncertainty"]

    for col in range(num_cols):
        res = all_results[col]
        axes[0, col].set_title(f"{res['name']}", fontsize=14, pad=10, fontweight='bold')
        axes[0, col].imshow(res["patch"], cmap='gray', vmin=v_min, vmax=v_max)
        axes[1, col].imshow(res["recon"], cmap='gray')
        for i in range(4): axes[2+i, col].imshow(res["samples"][i], cmap='gray')
        axes[6, col].imshow(res["epistemic"], cmap='magma', vmin=0, vmax=1.0)
        for row in range(num_rows):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if col == 0: axes[row, col].set_ylabel(labels[row], fontweight='bold', rotation=0, labelpad=60)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "single_patient_ood_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_best_checkpoint(net, checkpoint_dir, device="cpu"):
    best_file = os.path.join(checkpoint_dir, "model_latest.pth")
    if not os.path.exists(best_file): raise FileNotFoundError(f"Cannot find model_best.pth in {checkpoint_dir}")
    state_dict = torch.load(best_file, map_location=device)
    if 'model_state_dict' in state_dict: net.load_state_dict(state_dict['model_state_dict'])
    elif 'state_dict' in state_dict: net.load_state_dict(state_dict['state_dict'])
    else: net.load_state_dict(state_dict)
    print(f"[*] Successfully loaded optimal weights from: {best_file}")
    return net

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = "outputs/42_checkpoint_20260422_131855" 
    run_name = os.path.basename(os.path.normpath(checkpoint_dir))
    output_dir = os.path.join("test_results_ood", run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Lock all random mathematical operations
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_path  = 'data/test/'
    test_dataset = LIDC_IDRI_Folder(test_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    net = HierarchicalEvidentialUNet(input_channels=1, num_classes=2, latent_dims=[4,3,2,1], channels_per_block=[24, 48, 96, 192, 192, 192, 192, 192])
    net.to(device)
    net = load_best_checkpoint(net, checkpoint_dir, device=device)
    
    final_ged, final_dice, final_iou_rec, top_5_slices = evaluate_full_dataset(net, test_loader, device, output_dir, N_samples=16)
    plot_top_5_slices(top_5_slices, output_dir)
    print("\n[*] Running Multi-Patient Variance & Metric Analysis...")
    analyze_multiple_patients_performance(net, test_loader, device, output_dir, num_patients=5, N_samples=16) 
    print("\n[*] Running Dual Safety OOD Analysis...")
    analyze_single_patient_ood(net, test_loader, device, output_dir, N_samples=8)
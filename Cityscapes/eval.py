import os
import torch
import numpy as np
import scipy.ndimage as nd
import scipy.stats as stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_cityscapes_data import CityscapesColoringDataset, IGNORE_INDEX, NUM_CLASSES
from hierarchical_model_cityscapes import HierarchicalEvidentialUNetCityscapes

def append_h_cleanup(cluster_map, erode_size=5, majority_size=11):
    """Executes Appendix H: n x n erosion and m x m majority vote."""
    cleaned = np.zeros_like(cluster_map)
    instances = np.unique(cluster_map)
    instances = instances[instances > 0] # Ignore background (0)
    
    for inst_id in instances:
        mask = (cluster_map == inst_id)
        # 5x5 Erosion
        eroded = nd.binary_erosion(mask, structure=np.ones((erode_size, erode_size)))
        
        if eroded.sum() == 0:
            # Cluster eliminated, apply 11x11 majority vote
            # (In a full implementation, you compute the mode of the 11x11 neighborhood here)
            # For strict baseline diagnostic, we map frail clusters to background
            pass 
        else:
            cleaned[eroded] = inst_id
            
    return cleaned

def calculate_iou_matrix(pred_mask, gt_mask):
    """Calculates pairwise IoU between all predicted and GT instances."""
    pred_ids = np.unique(pred_mask)[1:] # Skip BG
    gt_ids = np.unique(gt_mask)[1:]
    
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, p_id in enumerate(pred_ids):
        p_mask = (pred_mask == p_id)
        for j, g_id in enumerate(gt_ids):
            g_mask = (gt_mask == g_id)
            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou_mat[i, j] = intersection / (union + 1e-8)
            
    return iou_mat, pred_ids, gt_ids

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Validation Data...")
    val_dataset = CityscapesColoringDataset(root_dir="/bhupender/datasets/Cityscapes", mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Loading Model Weights...")
    net = HierarchicalEvidentialUNetCityscapes(input_channels=3, num_classes=NUM_CLASSES).to(device)
    net.load_state_dict(torch.load("./outputs_cityscapes/model_final.pth", map_location=device))
    net.eval()
    
    total_samples = 32
    ious_50 = []

    with torch.no_grad():
        for images, multi_mask, _, info in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            gt_mask = multi_mask[:, 0, :, :].squeeze().cpu().numpy()
            
            # 1. Monte Carlo Prior Sampling
            mc_samples = []
            for _ in range(total_samples):
                logits = net.sample(images, use_mean=False) # MUST be False to test variance
                preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                mc_samples.append(preds)
            
            mc_samples = np.stack(mc_samples, axis=0) # [32, H, W]
            
            # 2. Variance Diagnostic (Check for Collapse)
            pixel_variance = np.var(mc_samples, axis=0)
            if pixel_variance.sum() == 0:
                print(f"\nWARNING: Zero variance detected on {info['path']}. Total Posterior Collapse.")
            
            # 3. Baseline Clustering (Simplified Mode / Hamming approximation)
            # In absence of full Algorithm 1 graph solver, the mode across samples serves as the consensus
            consensus_mask, _ = stats.mode(mc_samples, axis=0)
            consensus_mask = consensus_mask.squeeze()
            
            # 4. Appendix H Cleanup
            cleaned_mask = append_h_cleanup(consensus_mask)
            
            # 5. AP50 Calculation
            iou_mat, _, _ = calculate_iou_matrix(cleaned_mask, gt_mask)
            if iou_mat.size > 0:
                # Count True Positives (IoU >= 0.50)
                tp = (iou_mat.max(axis=1) >= 0.50).sum()
                precision = tp / len(iou_mat)
                ious_50.append(precision)

    print(f"\nMean AP50 Approximation: {np.mean(ious_50):.4f}")

if __name__ == "__main__":
    main()
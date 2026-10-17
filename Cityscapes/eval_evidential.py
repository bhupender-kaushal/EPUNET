import torch
import numpy as np
import torch.nn.functional as F
from skimage.segmentation import watershed
from skimage.measure import label
from scipy.ndimage import minimum_filter
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.ndimage as nd
import scipy.stats as stats

from load_cityscapes_data import CityscapesColoringDataset, NUM_CLASSES
from hierarchical_model_cityscapes import HierarchicalEvidentialUNetCityscapes


def signature_clustering(mc_samples):
    """
    mc_samples: [32, H, W] numpy array of integer predictions.
    Clusters pixels using their 32-dimensional permutation sequence.
    """
    _, H, W = mc_samples.shape
    
    # 1. Flatten spatial dimensions to create [H*W, 32] signature array
    signatures_flat = mc_samples.transpose(1, 2, 0).reshape(-1, 32)
    
    # 2. Extract unique 32-D signatures and map them back to an image grid
    # return_inverse provides the unique ID for every pixel's sequence
    unique_sigs, inverse_indices = np.unique(signatures_flat, axis=0, return_inverse=True)
    signature_map = inverse_indices.reshape(H, W)
    
    # 3. Connected Components
    # Isolates distinct cars that might theoretically share a signature
    labeled_map, _ = nd.label(signature_map)
    
    # 4. Background Masking
    # The background is the mode across the 32 samples (Class 0)
    mode_mask, _ = stats.mode(mc_samples, axis=0)
    bg_mask = (mode_mask.squeeze() == 0)
    labeled_map[bg_mask] = 0
    
    # 5. Appendix H Cleanup
    # Boundary pixels will have flickering, unique signatures. 
    # Erosion deletes the noisy halo; majority vote (not fully implemented here, 
    # but mimicked via connectivity) restores the boundary.
    cleaned_mask = append_h_cleanup(labeled_map)
    
    return cleaned_mask


def evidential_watershed_projected(alpha, expected_p):
    # 1. Binary Dirichlet Projection
    alpha_bg = alpha[0, :, :]
    alpha_fg = np.sum(alpha[1:, :, :], axis=0)
    S_bin = alpha_bg + alpha_fg
    
    # 2. Binary Epistemic Variance (Topology only, ignores color permutations)
    fg_var = (alpha_fg * alpha_bg) / ((S_bin ** 2) * (S_bin + 1.0))
    fg_prob = np.sum(expected_p[1:], axis=0) 
    
    # 3. Define the topological landscape
    energy_landscape = -fg_prob + fg_var 
    
    # 4. Locate seeds (Reduced filter size to 5 to protect distant, small cars)
    core_mask = (fg_prob > 0.6) & (fg_var < np.percentile(fg_var, 50))
    local_min = minimum_filter(energy_landscape, size=5) == energy_landscape
    seeds = label(core_mask & local_min)
    
    # 5. Evidential Watershed Flooding
    instance_mask = watershed(energy_landscape, seeds, mask=(fg_prob > 0.4))
    
    return instance_mask


def calculate_ap50(pred_mask, gt_mask, expected_p):
    """
    Computes Average Precision at IoU=0.50 threshold using Evidential confidence.
    """
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0] # Ignore background
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0

    # 1. Calculate Confidence for each prediction
    # Foreground probability is the sum of classes 1-5
    fg_prob = np.sum(expected_p[1:], axis=0) 
    
    scored_preds = []
    for p_id in pred_ids:
        p_mask = (pred_mask == p_id)
        confidence = fg_prob[p_mask].mean()
        scored_preds.append((confidence, p_id, p_mask))
        
    # Sort predictions by descending confidence
    scored_preds.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Build IoU Matrix
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, (_, p_id, p_mask) in enumerate(scored_preds):
        for j, g_id in enumerate(gt_ids):
            g_mask = (gt_mask == g_id)
            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou_mat[i, j] = intersection / (union + 1e-8)
            
    # 3. Greedy Matching at 0.50 Threshold
    tp = 0
    fp = 0
    matched_gt = set()
    
    for i in range(len(scored_preds)):
        best_gt_idx = np.argmax(iou_mat[i])
        best_iou = iou_mat[i, best_gt_idx]
        
        if best_iou >= 0.50 and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = CityscapesColoringDataset(root_dir="/bhupender/datasets/Cityscapes", mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    net = HierarchicalEvidentialUNetCityscapes(input_channels=3, num_classes=NUM_CLASSES).to(device)
    net.load_state_dict(torch.load("./outputs_cityscapes/model_final.pth", map_location=device))
    net.eval()
    
    ious_50 = []

    with torch.no_grad():
        for images, multi_mask, _, _ in tqdm(val_loader, desc="Evidential Eval"):
            images = images.to(device)
            gt_mask = multi_mask[:, 0, :, :].squeeze().cpu().numpy()
            
        #     # 1. Single Deterministic Forward Pass (Prior Mean)
        #     prior_out = net.prior(images, use_mean=True)
        #     logits = net.f_comb(prior_out['encoder_features'], prior_out['decoder_features'])
            
        #    # 2. Dirichlet Mathematics
        #     evidence = torch.clamp(F.softplus(logits), max=200.0)
        #     alpha = evidence + 1.0
        #     S = torch.sum(alpha, dim=1, keepdim=True)
            
        #     expected_p = (alpha / S).squeeze().cpu().numpy()
        #     alpha_np = alpha.squeeze().cpu().numpy()
            
        #     # 3. Apply the Projected Watershed
        #     pred_instances = evidential_watershed_projected(alpha_np, expected_p)
            
        #     # 4. AP Metric Logic
        #     ap50 = calculate_ap50(pred_instances, gt_mask, expected_p)
        #     ious_50.append(ap50)
        # 1. Monte Carlo Prior Sampling
            mc_samples = []
            for _ in range(32):
                logits = net.sample(images, use_mean=False) 
                preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                mc_samples.append(preds)
            
            mc_samples = np.stack(mc_samples, axis=0) # [32, H, W]
            
            # 2. Cryptographic Signature Clustering
            pred_instances = signature_clustering(mc_samples)
            
            # 3. Confidence Extraction (Using Mode for Foreground Prob)
            # Standardize expected_p calculation for the AP50 function
            mode_preds, _ = nd.stats.mode(mc_samples, axis=0)
            expected_p = np.zeros((NUM_CLASSES, images.size(2), images.size(3)))
            for c in range(NUM_CLASSES):
                expected_p[c] = (mode_preds.squeeze() == c).astype(float)
            
            # 4. AP Metric Logic
            ap50 = calculate_ap50(pred_instances, gt_mask, expected_p)
            ious_50.append(ap50)

    print(f"\nEvidential AP50 Score: {np.mean(ious_50):.4f}")

if __name__ == "__main__":
    main()
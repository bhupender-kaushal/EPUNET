# import os
# import glob
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import scipy.sparse
# from scipy.sparse.csgraph import connected_components
# from scipy import ndimage
# from skimage.segmentation import watershed
# from skimage.color import label2rgb
# import matplotlib.pyplot as plt

# from scipy import ndimage
# from skimage.segmentation import watershed



# from load_snemi3d_data import SNEMI3DPatches, NUM_CLASSES, IGNORE_INDEX, PATCH_SIZE
# from hierarchical_model_snemi3d import HierarchicalEvidentialUNetSNEMI3D


# @torch.no_grad()
# def evidential_watershed_instance_map(net, patch, device, n_samples=16, variance_threshold=0.05):
#     """
#     Executes a topological watershed using Dirichlet variance as the boundary constraint.
#     """
#     net.eval()
#     mc_predictions = []
#     mc_variances = []
    
#     prior_out = net.prior(patch)
#     enc_cache = prior_out['encoder_features']
    
#     # 1. Stochastic MC Sampling for Epistemic Variance
#     for _ in range(n_samples):
#         z_sample = [dist.rsample() for dist in prior_out['distributions']]
#         prior_dec = net.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
#         logits = net.f_comb(enc_cache, prior_dec)
        
#         evidence = torch.clamp(F.softplus(logits), max=200.0)
#         alpha = evidence + 1.0
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         expected_p = alpha / S
        
#         _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
#         # Calculate precise Dirichlet Variance
#         alpha_c = torch.gather(alpha, 1, pred_cls)
#         var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
#         mc_predictions.append(pred_cls)
#         mc_variances.append(var_c)

#     # 2. Consensus Aggregation
#     mc_predictions = torch.cat(mc_predictions, dim=1) 
#     mc_variances = torch.cat(mc_variances, dim=1) 
    
#     consensus_pred, _ = torch.mode(mc_predictions, dim=1, keepdim=True)
#     mean_variance = mc_variances.mean(dim=1, keepdim=True)
    
#     # 3. Isolate High-Confidence Core Seeds
#     # Must be a foreground class (>0) AND have low variance
#     core_mask = ((consensus_pred > 0) & (mean_variance < variance_threshold)).squeeze().cpu().numpy()
    
#     # 4. Connected Components on the strict cores
#     labeled_cores, _ = ndimage.label(core_mask)
    
#     # 5. Topological Watershed
#     # Expand the safe cores into the high-variance membrane territories
#     valid_em_mask = (patch[0, 0].cpu().numpy() > 0)
    
#     # Compute distance from the cores to direct the watershed expansion
#     distance = ndimage.distance_transform_edt(labeled_cores == 0)
    
#     final_instance_mask = watershed(distance, labeled_cores, mask=valid_em_mask)
    
#     return final_instance_mask


# def load_latest_checkpoint(net, checkpoint_dir, device="cpu", prefer="model_best.pth"):
#     preferred = os.path.join(checkpoint_dir, prefer)
#     if os.path.exists(preferred):
#         path = preferred
#     else:
#         candidates = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
#         if not candidates:
#             raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
#         path = max(candidates, key=os.path.getctime)
#     state = torch.load(path, map_location=device)
#     net.load_state_dict(state.get("state_dict", state))
#     print(f"Loaded checkpoint: {path} (epoch {state.get('epoch', '?')})")
#     return net

# @torch.no_grad()
# def evaluate_deterministic_performance(net, loader, device):
#     net.eval()
#     inter, union, tp, fp, fn = 0, 0, 0, 0, 0
#     total_recon_error = 0.0
#     num_batches = 0
    
#     for patch, mask_labels, _, _ in loader:
#         patch = patch.to(device)
#         mask_labels = mask_labels.to(device).long().squeeze(1)

#         # Uses the Prior to guess (Standard Inference)
#         logits = net.sample(patch, use_mean=True)
        
#         recon_error = F.cross_entropy(logits, mask_labels, ignore_index=IGNORE_INDEX)
#         total_recon_error += recon_error.item()
#         num_batches += 1
        
#         probs = F.softplus(logits) + 1.0
#         probs = probs / probs.sum(dim=1, keepdim=True)
#         pred = torch.argmax(probs, dim=1) 

#         valid = mask_labels != IGNORE_INDEX
#         pred_fg = (pred != 0) & valid
#         true_fg = (mask_labels != 0) & valid

#         inter += (pred_fg & true_fg).sum().item()
#         union += (pred_fg | true_fg).sum().item()
#         tp += (pred_fg & true_fg).sum().item()
#         fp += (pred_fg & ~true_fg).sum().item()
#         fn += (~pred_fg & true_fg).sum().item()

#     iou = inter / max(union, 1)
#     precision = tp / max(tp + fp, 1)
#     recall = tp / max(tp + fn, 1)
#     f1 = 2 * precision * recall / max(precision + recall, 1e-8)
#     avg_recon = total_recon_error / max(num_batches, 1)
    
#     return {"fg_iou": iou, "fg_precision": precision, "fg_recall": recall, "fg_f1": f1, "prior_cross_entropy": avg_recon}

# @torch.no_grad()
# def evaluate_posterior_reconstruction_iou(net, loader, device):
#     """
#     Evaluates Appendix B.2 Reconstruction Fidelity.
#     IoU_rec(Y, Y') where Y' = S(X, mu_post(X, Y)) using NAN-mean logic.
#     """
#     net.eval()
#     per_image_ious = []
    
#     for patch, mask_labels, _, _ in loader:
#         patch = patch.to(device)
#         mask_labels = mask_labels.to(device).long().squeeze(1)
        
#         # 1. Map to One-Hot
#         mask_onehot = F.one_hot(torch.clamp(mask_labels, min=0, max=NUM_CLASSES-1), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        
#         # 2. Extract Prior Features
#         enc_features = net.prior(patch)['encoder_features']
        
#         # 3. Extract Posterior Distributions
#         # EXACT MATCH: Must be [mask_onehot, patch] as defined in forward_train_step
#         posterior_input = torch.cat([mask_onehot, patch], dim=1)
#         q_distributions = net.posterior(posterior_input)['distributions']
        
#         # 4. Extract Latent Mean and Decode
#         # EXACT MATCH: PyTorch Independent(Normal) uses .mean
#         z_mu = [dist.mean for dist in q_distributions]
#         prior_dec = net.prior(encoder_features=enc_features, z_q=z_mu)['decoder_features']
#         logits = net.f_comb(enc_features, prior_dec)
        
#         # 5. Evaluate Categorical Probabilities
#         probs = F.softplus(logits) + 1.0
#         pred_cls = torch.argmax(probs / probs.sum(dim=1, keepdim=True), dim=1)
        
#         # 6. NAN-Mean Execution
#         batch_size = patch.size(0)
#         for i in range(batch_size):
#             valid_mask = mask_labels[i] != IGNORE_INDEX
            
#             gt_fg = (mask_labels[i] != 0) & valid_mask
#             pred_fg = (pred_cls[i] != 0) & valid_mask
            
#             gt_total = gt_fg.sum().float()
#             pred_total = pred_fg.sum().float()
            
#             if gt_total == 0:
#                 if pred_total == 0:
#                     per_image_ious.append(np.nan)
#                 else:
#                     per_image_ious.append(0.0)
#             else:
#                 intersection = (gt_fg & pred_fg).sum().float()
#                 union = (gt_fg | pred_fg).sum().float()
#                 iou = intersection / max(union, 1.0)
#                 per_image_ious.append(iou.item())
                
#     return np.nanmean(per_image_ious)

# @torch.no_grad()
# def evaluate_epistemic_uncertainty(net, loader, device, max_batches=None):
#     net.eval()
#     total_u, total_px = 0.0, 0
#     for i, (patch, mask_labels, _, _) in enumerate(loader):
#         if max_batches is not None and i >= max_batches:
#             break
#         patch = patch.to(device)
#         mask_labels = mask_labels.to(device).long()

#         logits = net.sample(patch, use_mean=True)
#         evidence = F.softplus(logits)
#         alpha = evidence + 1.0
#         S = alpha.sum(dim=1, keepdim=True)
#         u = (NUM_CLASSES / S).squeeze(1)

#         valid = mask_labels.squeeze(1) != IGNORE_INDEX
#         total_u += u[valid].sum().item()
#         total_px += valid.sum().item()
#     return total_u / max(total_px, 1)

# def greedy_hamming_clustering(masked_preds, threshold=8, ignore_index=255):
#     S, H, W = masked_preds.shape
#     flat_preds = masked_preds.view(S, -1).t()
    
#     N_pixels = flat_preds.size(0)
#     instance_map = torch.zeros(N_pixels, dtype=torch.long, device=masked_preds.device)
    
#     valid_pixel_mask = (flat_preds != ignore_index).any(dim=1)
#     unassigned_indices = torch.nonzero(valid_pixel_mask).squeeze(1)
    
#     current_instance_id = 1
    
#     while unassigned_indices.numel() > 0:
#         seed_idx = unassigned_indices[0]
#         seed_signature = flat_preds[seed_idx].unsqueeze(0) 
        
#         unassigned_signatures = flat_preds[unassigned_indices] 
#         valid_comparisons = (unassigned_signatures != ignore_index) & (seed_signature != ignore_index)
#         mismatches = (unassigned_signatures != seed_signature) & valid_comparisons
#         distances = mismatches.sum(dim=1) 
        
#         cluster_mask = distances <= threshold
#         cluster_indices = unassigned_indices[cluster_mask]
        
#         instance_map[cluster_indices] = current_instance_id
#         current_instance_id += 1
        
#         unassigned_indices = unassigned_indices[~cluster_mask]
        
#     return instance_map.view(H, W)

# @torch.no_grad()
# def evidential_watershed_instance_map(net, patch, device, n_samples=16, variance_threshold=0.05):
#     net.eval()
#     mc_predictions = []
#     mc_variances = []
    
#     prior_out = net.prior(patch)
#     enc_cache = prior_out['encoder_features']
    
#     # 1. Stochastic MC Sampling for Epistemic Variance
#     for _ in range(n_samples):
#         z_sample = [dist.rsample() for dist in prior_out['distributions']]
#         prior_dec = net.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
#         logits = net.f_comb(enc_cache, prior_dec)
        
#         evidence = torch.clamp(F.softplus(logits), max=200.0)
#         alpha = evidence + 1.0
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         expected_p = alpha / S
        
#         _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
#         # Calculate precise Dirichlet Variance
#         alpha_c = torch.gather(alpha, 1, pred_cls)
#         var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
#         mc_predictions.append(pred_cls)
#         mc_variances.append(var_c)

#     # 2. Consensus Aggregation
#     mc_predictions = torch.cat(mc_predictions, dim=1) 
#     mc_variances = torch.cat(mc_variances, dim=1) 
    
#     # Find pixels consistently predicted as the same class, and average their variance
#     consensus_pred, _ = torch.mode(mc_predictions, dim=1, keepdim=True)
#     mean_variance = mc_variances.mean(dim=1, keepdim=True)
    
#     # 3. Isolate High-Confidence Core Seeds
#     # Must be a foreground class AND have low variance
#     core_mask = ((consensus_pred > 0) & (mean_variance < variance_threshold)).squeeze().cpu().numpy()
    
#     # 4. Connected Components on the strict cores
#     # This guarantees adjacent cells don't touch
#     labeled_cores, num_features = ndimage.label(core_mask)
    
#     # 5. Topological Watershed
#     # Expand the safe cores into the high-variance membrane territories
#     # The mask ensures the watershed doesn't spill into true EM background
#     valid_em_mask = (patch[0, 0].cpu().numpy() > 0)
    
#     # Compute distance from the cores to direct the watershed expansion
#     distance = ndimage.distance_transform_edt(labeled_cores == 0)
    
#     final_instance_mask = watershed(distance, labeled_cores, mask=valid_em_mask)
    
#     return final_instance_mask


# from scipy.ndimage import binary_erosion
# from scipy.stats import mode

# def paper_appendix_h_post_processing(instance_map, n=5, m=11):
#     """
#     Executes the exact morphological post-processing defined in Kohl et al. (2018) Appendix H.
#     instance_map: 2D numpy array of clustered instance IDs.
#     """
#     processed_map = instance_map.copy()
#     cluster_ids = np.unique(instance_map)
    
#     # 1. Define the n x n structure element for erosion
#     struct_el = np.ones((n, n), dtype=bool)
    
#     clusters_to_remove = []
    
#     # 2. Erosion Check: Identify tiny/thin boundary clusters
#     for c in cluster_ids:
#         if c == 0: # Do not erode the background
#             continue
            
#         mask = (instance_map == c)
#         eroded = binary_erosion(mask, structure=struct_el)
        
#         # If the cluster does not survive the erosion, flag it for deletion
#         if not eroded.any():
#             clusters_to_remove.append(c)
            
#     if not clusters_to_remove:
#         return processed_map # No noise clusters found, return as-is
        
#     # 3. Prepare for Majority Vote
#     # Pad the map so the m x m window doesn't go out of bounds at the image edges
#     pad_w = m // 2
#     padded_map = np.pad(instance_map, pad_width=pad_w, mode='constant', constant_values=0)
    
#     # Create a boolean mask of all pixels that belong to the condemned clusters
#     remove_mask = np.isin(instance_map, clusters_to_remove)
#     y_coords, x_coords = np.where(remove_mask)
    
#     # 4. m x m Majority Vote Painting
#     for y, x in zip(y_coords, x_coords):
#         current_label = instance_map[y, x]
        
#         # Extract the m x m window from the padded array (offset by pad_w)
#         window = padded_map[y : y + m, x : x + m]
        
#         # Rule: Ignore background (0) and the condemned cluster's own label
#         valid_mask = (window != 0) & (window != current_label)
#         valid_neighbors = window[valid_mask]
        
#         if len(valid_neighbors) > 0:
#             # Paint in the majority label
#             majority_label = mode(valid_neighbors, keepdims=True)[0][0]
#             processed_map[y, x] = majority_label
#         else:
#             # SNEMI3D specific rule: If 0 valid neighbors, keep original label
#             pass 
            
#     return processed_map

# @torch.no_grad()
# def evidential_hamming_instance_map(net, patch, device, n_samples=16, variance_threshold=0.10, distance_threshold=8):
#     net.eval()
#     mc_predictions = []
#     mc_valid_masks = []
    
#     prior_out = net.prior(patch)
#     enc_cache = prior_out['encoder_features']
    
#     for _ in range(n_samples):
#         z_sample = [dist.rsample() for dist in prior_out['distributions']]
#         prior_dec = net.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
#         logits = net.f_comb(enc_cache, prior_dec)
        
#         evidence = torch.clamp(F.softplus(logits), max=200.0)
#         alpha = evidence + 1.0
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         expected_p = alpha / S
        
#         bg_sensitivity = 3.0
#         expected_p[:, 0, :, :] *= bg_sensitivity
        
#         _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
#         alpha_c = torch.gather(alpha, 1, pred_cls)
#         var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
#         valid_mask = (var_c < variance_threshold).float()
        
#         mc_predictions.append(pred_cls)
#         mc_valid_masks.append(valid_mask)

#     mc_predictions = torch.cat(mc_predictions, dim=1) 
#     mc_valid_masks = torch.cat(mc_valid_masks, dim=1) 
    
#     final_valid_mask = (mc_valid_masks.mean(dim=1, keepdim=True) > 0.5).long()
#     masked_predictions = torch.where(final_valid_mask == 1, mc_predictions, torch.tensor(IGNORE_INDEX, device=device))
    
#     core_instances = greedy_hamming_clustering(masked_predictions[0], threshold=distance_threshold)
#     core_instances = core_instances.cpu().numpy()
    
#     distance = ndimage.distance_transform_edt(core_instances > 0)
#     final_instance_mask = watershed(-distance, core_instances, mask=(patch[0, 0].cpu().numpy() > 0))
    
#     return final_instance_mask

# def _comb2(x):
#     return x * (x - 1) / 2.0

# def accumulate_rand_terms(pred_instance, true_instance, running):
#     mask = true_instance != 0
#     p = pred_instance[mask].ravel()
#     t = true_instance[mask].ravel()
#     if p.size == 0: return

#     p_ids, p_inv = np.unique(p, return_inverse=True)
#     t_ids, t_inv = np.unique(t, return_inverse=True)
#     n_p, n_t = len(p_ids), len(t_ids)

#     contingency = scipy.sparse.coo_matrix(
#         (np.ones_like(p_inv, dtype=np.int64), (t_inv, p_inv)), shape=(n_t, n_p)
#     ).toarray()

#     running["sum_comb_c"] += _comb2(contingency).sum()
#     running["sum_comb_k"] += _comb2(contingency.sum(axis=1)).sum()
#     running["sum_comb_l"] += _comb2(contingency.sum(axis=0)).sum()

# def finalize_adapted_rand(running):
#     sum_comb_c, sum_comb_k, sum_comb_l = running["sum_comb_c"], running["sum_comb_k"], running["sum_comb_l"]
#     precision = sum_comb_c / sum_comb_l if sum_comb_l > 0 else 0.0
#     recall = sum_comb_c / sum_comb_k if sum_comb_k > 0 else 0.0
#     f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
#     return {"adapted_rand_error": 1.0 - f_score, "precision": precision, "recall": recall, "f_score": f_score}

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_dir", default="/bhupender/2026/PUNET/new_punet/SNIMED/outputs_snemi3d/42_checkpoint_20260805_162559")
#     parser.add_argument("--images_path", type=str, default="/bhupender/datasets/SNEMI3D/image/train-input.tif")
#     parser.add_argument("--labels_path", type=str, default="/bhupender/datasets/SNEMI3D/seg/train-labels.tif")  
#     parser.add_argument("--n_instance_samples", type=int, default=16)
#     parser.add_argument("--max_instance_patches", type=int, default=None)
#     parser.add_argument("--save_vis_dir", type=str, default="segmentation_overlays")
#     args = parser.parse_args()

#     os.makedirs(args.save_vis_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     net = HierarchicalEvidentialUNetSNEMI3D(
#         input_channels=1, 
#         num_classes=NUM_CLASSES,
#         latent_dims=(40, 25, 15, 5),
#         channels_per_block=(32, 64, 128, 256, 256, 256), 
#         num_resblocks=2, 
#     ).to(device)
    
#     net = load_latest_checkpoint(net, args.checkpoint_dir, device=device)

#     test_dataset = SNEMI3DPatches(args.images_path, args.labels_path, mode="test")
#     test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

#     # 1. Deterministic Prior Metrics
#     fg_metrics = evaluate_deterministic_performance(net, test_loader, device)
#     print("\n--- Deterministic Inference (Prior) ---")
#     for k, v in fg_metrics.items():
#         print(f"  {k}: {v:.4f}")

#     # 2. Upper Bound Posterior Reconstruction (New B.2 Paper Metric)
#     print("\n--- Reconstruction Fidelity (Posterior) ---")
#     try:
#         posterior_iou = evaluate_posterior_reconstruction_iou(net, test_loader, device)
#         print(f"  reconstruction_iou (NAN-mean): {posterior_iou:.4f}")
#     except NotImplementedError as e:
#         print(f"  [Error] {e}")

#     # 3. Uncertainty Metrics
#     uncertainty = evaluate_epistemic_uncertainty(net, test_loader, device)
#     print(f"\nMean epistemic uncertainty (test set): {uncertainty:.4f}")

#     # 4. Instance Clustering & Image Overlay Generation
#     print(f"\n--- Instance segmentation via {args.n_instance_samples}-sample Evidential Hamming Clustering ---")
#     running = {"sum_comb_c": 0.0, "sum_comb_k": 0.0, "sum_comb_l": 0.0}
#     n_scored = 0
    
#     for idx, (z, y0, x0) in enumerate(test_dataset.anchors):
#         if args.max_instance_patches is not None and idx >= args.max_instance_patches:
#             break
            
#         img_patch = test_dataset.images[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
#         true_instance = test_dataset.labels[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] 

#         img_t = torch.from_numpy(img_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
#         # pred_instance = evidential_hamming_instance_map(net, img_t, device, n_samples=args.n_instance_samples)
#         pred_instance = evidential_watershed_instance_map(
#                 net=net, 
#                 patch=img_t, 
#                 device=device, 
#                 n_samples=args.n_instance_samples, 
#                 variance_threshold=0.05
#             )

#         pred_instance = paper_appendix_h_post_processing(pred_instance, n=5, m=11)

#         accumulate_rand_terms(pred_instance, true_instance, running)
        
#         # --- GENERATE & SAVE OVERLAYS ---
#         if n_scored < 5:
#             with torch.no_grad():
#                 logits_det = net.sample(img_t, use_mean=True)
#                 evidence = F.softplus(logits_det)
#                 alpha = evidence + 1.0
#                 S = torch.sum(alpha, dim=1, keepdim=True)
                
#                 pred_cls_det = torch.argmax(alpha / S, dim=1).squeeze().cpu().numpy()
#                 uncertainty_map = (NUM_CLASSES / S).squeeze().cpu().numpy()

#             base_img = img_patch.astype(np.float32) / 255.0
#             pred_overlay = label2rgb(pred_instance, image=base_img, bg_label=0, alpha=0.4)
#             true_overlay = label2rgb(true_instance, image=base_img, bg_label=0, alpha=0.4)
            
#             fig, axs = plt.subplots(1, 5, figsize=(25, 5))
            
#             axs[0].imshow(base_img, cmap='gray')
#             axs[0].set_title(f'Original EM [Patch {n_scored}]')
            
#             axs[1].imshow(true_overlay)
#             axs[1].set_title('Ground Truth Instances')
            
#             axs[2].imshow(pred_cls_det, cmap='tab20', vmin=0, vmax=15)
#             axs[2].set_title('Raw 16-Class Reconstruction')
            
#             im_u = axs[3].imshow(uncertainty_map, cmap='magma')
#             axs[3].set_title('Epistemic Uncertainty Map')
#             plt.colorbar(im_u, ax=axs[3], fraction=0.046, pad=0.04)
            
#             axs[4].imshow(pred_overlay)
#             axs[4].set_title('Final Instances (Hamming)')
            
#             for ax in axs: 
#                 ax.axis('off')
                
#             plt.tight_layout()
#             save_path = os.path.join(args.save_vis_dir, f"patch_reconstruction_{n_scored:03d}.png")
#             plt.savefig(save_path, dpi=150)
#             plt.close()

#         n_scored += 1

#     rand_metrics = finalize_adapted_rand(running)
#     print(f"  Scored {n_scored} test patches")
#     for k, v in rand_metrics.items():
#         print(f"  {k}: {v:.4f}")

# if __name__ == "__main__":
#     main()



import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.sparse
from scipy.sparse.csgraph import connected_components
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import heapq

from load_snemi3d_data import SNEMI3DPatches, NUM_CLASSES, IGNORE_INDEX, PATCH_SIZE
from hierarchical_model_snemi3d import HierarchicalEvidentialUNetSNEMI3D

def load_latest_checkpoint(net, checkpoint_dir, device="cpu", prefer="model_best.pth"):
    preferred = os.path.join(checkpoint_dir, prefer)
    if os.path.exists(preferred):
        path = preferred
    else:
        candidates = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        path = max(candidates, key=os.path.getctime)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state.get("state_dict", state))
    print(f"Loaded checkpoint: {path} (epoch {state.get('epoch', '?')})")
    return net

@torch.no_grad()
def evaluate_deterministic_performance(net, loader, device):
    net.eval()
    inter, union, tp, fp, fn = 0, 0, 0, 0, 0
    total_recon_error = 0.0
    num_batches = 0
    
    for patch, mask_labels, _, _ in loader:
        patch = patch.to(device)
        mask_labels = mask_labels.to(device).long().squeeze(1)

        logits = net.sample(patch, use_mean=True)
        recon_error = F.cross_entropy(logits, mask_labels, ignore_index=IGNORE_INDEX)
        total_recon_error += recon_error.item()
        num_batches += 1
        
        probs = F.softplus(logits) + 1.0
        probs = probs / probs.sum(dim=1, keepdim=True)
        pred = torch.argmax(probs, dim=1) 

        valid = mask_labels != IGNORE_INDEX
        pred_fg = (pred != 0) & valid
        true_fg = (mask_labels != 0) & valid

        inter += (pred_fg & true_fg).sum().item()
        union += (pred_fg | true_fg).sum().item()
        tp += (pred_fg & true_fg).sum().item()
        fp += (pred_fg & ~true_fg).sum().item()
        fn += (~pred_fg & true_fg).sum().item()

    iou = inter / max(union, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_recon = total_recon_error / max(num_batches, 1)
    
    return {"fg_iou": iou, "fg_precision": precision, "fg_recall": recall, "fg_f1": f1, "prior_cross_entropy": avg_recon}

@torch.no_grad()
def evaluate_posterior_reconstruction_iou(net, loader, device):
    net.eval()
    per_image_ious = []
    
    for patch, mask_labels, _, _ in loader:
        patch = patch.to(device)
        mask_labels = mask_labels.to(device).long().squeeze(1)
        
        mask_onehot = F.one_hot(torch.clamp(mask_labels, min=0, max=NUM_CLASSES-1), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        enc_features = net.prior(patch)['encoder_features']
        posterior_input = torch.cat([mask_onehot, patch], dim=1)
        q_distributions = net.posterior(posterior_input)['distributions']
        
        z_mu = [dist.mean for dist in q_distributions]
        prior_dec = net.prior(encoder_features=enc_features, z_q=z_mu)['decoder_features']
        logits = net.f_comb(enc_features, prior_dec)
        
        probs = F.softplus(logits) + 1.0
        pred_cls = torch.argmax(probs / probs.sum(dim=1, keepdim=True), dim=1)
        
        batch_size = patch.size(0)
        for i in range(batch_size):
            valid_mask = mask_labels[i] != IGNORE_INDEX
            gt_fg = (mask_labels[i] != 0) & valid_mask
            pred_fg = (pred_cls[i] != 0) & valid_mask
            
            gt_total = gt_fg.sum().float()
            pred_total = pred_fg.sum().float()
            
            if gt_total == 0:
                if pred_total == 0: per_image_ious.append(np.nan)
                else: per_image_ious.append(0.0)
            else:
                intersection = (gt_fg & pred_fg).sum().float()
                union = (gt_fg | pred_fg).sum().float()
                iou = intersection / max(union, 1.0)
                per_image_ious.append(iou.item())
                
    return np.nanmean(per_image_ious)

@torch.no_grad()
def evaluate_epistemic_uncertainty(net, loader, device, max_batches=None):
    net.eval()
    total_u, total_px = 0.0, 0
    for i, (patch, mask_labels, _, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        patch = patch.to(device)
        mask_labels = mask_labels.to(device).long()

        logits = net.sample(patch, use_mean=True)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        u = (NUM_CLASSES / S).squeeze(1)

        valid = mask_labels.squeeze(1) != IGNORE_INDEX
        total_u += u[valid].sum().item()
        total_px += valid.sum().item()
    return total_u / max(total_px, 1)

@torch.no_grad()
def evidential_watershed_instance_map(net, patch, device, n_samples=16, variance_threshold=0.05):
    net.eval()
    mc_predictions = []
    mc_variances = []
    
    prior_out = net.prior(patch)
    enc_cache = prior_out['encoder_features']
    
    for _ in range(n_samples):
        z_sample = [dist.rsample() for dist in prior_out['distributions']]
        prior_dec = net.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
        logits = net.f_comb(enc_cache, prior_dec)
        
        evidence = torch.clamp(F.softplus(logits), max=200.0)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        expected_p = alpha / S
        
        _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
        alpha_c = torch.gather(alpha, 1, pred_cls)
        var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
        mc_predictions.append(pred_cls)
        mc_variances.append(var_c)

    mc_predictions = torch.cat(mc_predictions, dim=1) 
    mc_variances = torch.cat(mc_variances, dim=1) 
    
    consensus_pred, _ = torch.mode(mc_predictions, dim=1, keepdim=True)
    mean_variance = mc_variances.mean(dim=1, keepdim=True)
    
    core_mask = ((consensus_pred > 0) & (mean_variance < variance_threshold)).squeeze().cpu().numpy()
    labeled_cores, num_features = ndimage.label(core_mask)
    
    valid_em_mask = (patch[0, 0].cpu().numpy() > 0)
    distance = ndimage.distance_transform_edt(labeled_cores == 0)
    final_instance_mask = watershed(distance, labeled_cores, mask=valid_em_mask)
    
    return final_instance_mask

from scipy.ndimage import binary_erosion
from scipy.stats import mode

def paper_appendix_h_post_processing(instance_map, n=5, m=11):
    processed_map = instance_map.copy()
    cluster_ids = np.unique(instance_map)
    struct_el = np.ones((n, n), dtype=bool)
    clusters_to_remove = []
    
    for c in cluster_ids:
        if c == 0: continue
        mask = (instance_map == c)
        eroded = binary_erosion(mask, structure=struct_el)
        if not eroded.any(): clusters_to_remove.append(c)
            
    if not clusters_to_remove: return processed_map
        
    pad_w = m // 2
    padded_map = np.pad(instance_map, pad_width=pad_w, mode='constant', constant_values=0)
    remove_mask = np.isin(instance_map, clusters_to_remove)
    y_coords, x_coords = np.where(remove_mask)
    
    for y, x in zip(y_coords, x_coords):
        current_label = instance_map[y, x]
        window = padded_map[y : y + m, x : x + m]
        valid_mask = (window != 0) & (window != current_label)
        valid_neighbors = window[valid_mask]
        
        if len(valid_neighbors) > 0:
            majority_label = mode(valid_neighbors, keepdims=True)[0][0]
            processed_map[y, x] = majority_label
            
    return processed_map

def _comb2(x):
    return x * (x - 1) / 2.0

def accumulate_rand_terms(pred_instance, true_instance, running):
    mask = true_instance != 0
    p = pred_instance[mask].ravel()
    t = true_instance[mask].ravel()
    if p.size == 0: return

    p_ids, p_inv = np.unique(p, return_inverse=True)
    t_ids, t_inv = np.unique(t, return_inverse=True)
    n_p, n_t = len(p_ids), len(t_ids)

    contingency = scipy.sparse.coo_matrix(
        (np.ones_like(p_inv, dtype=np.int64), (t_inv, p_inv)), shape=(n_t, n_p)
    ).toarray()

    running["sum_comb_c"] += _comb2(contingency).sum()
    running["sum_comb_k"] += _comb2(contingency.sum(axis=1)).sum()
    running["sum_comb_l"] += _comb2(contingency.sum(axis=0)).sum()

def finalize_adapted_rand(running):
    sum_comb_c = running["sum_comb_c"]
    sum_comb_k = running["sum_comb_k"] 
    sum_comb_l = running["sum_comb_l"]
    precision = sum_comb_c / sum_comb_l if sum_comb_l > 0 else 0.0
    recall = sum_comb_c / sum_comb_k if sum_comb_k > 0 else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"adapted_rand_error": 1.0 - f_score, "precision": precision, "recall": recall, "f_score": f_score}

@torch.no_grad()
def render_comprehensive_patch_diagnostics(net, test_dataset, idx, rank, score, save_base_dir, device, n_samples=16):
    # 1. Raw Memory Extraction (Bypassing __getitem__ dimension injections)
    z, y0, x0 = test_dataset.anchors[idx]
    img_patch = test_dataset.images[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
    true_instance = test_dataset.labels[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] 
    
    # 2. Format Tensors for the Network
    img_t = torch.from_numpy(img_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(true_instance).unsqueeze(0).to(device).long()
    
    # 3. Format Arrays for skimage/matplotlib
    base_img = img_patch.astype(np.float32) / 255.0
    
    # Create the dedicated directory for this specific tile
    patch_dir = os.path.join(save_base_dir, f"rank_{rank:02d}_patch_{idx:04d}_fscore_{score:.4f}")
    samples_dir = os.path.join(patch_dir, "16_prior_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    true_overlay = label2rgb(true_instance, image=base_img, bg_label=0, alpha=0.4)

    # ---------------------------------------------------------
    # 1. Posterior Reconstruction Upper-Bound (Cheating with GT)
    # ---------------------------------------------------------
    mask_onehot = F.one_hot(torch.clamp(mask_t, min=0, max=NUM_CLASSES-1), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    enc_features = net.prior(img_t)['encoder_features']
    posterior_input = torch.cat([mask_onehot, img_t], dim=1)
    q_distributions = net.posterior(posterior_input)['distributions']
    
    z_mu = [dist.mean for dist in q_distributions]
    prior_dec = net.prior(encoder_features=enc_features, z_q=z_mu)['decoder_features']
    logits_post = net.f_comb(enc_features, prior_dec)
    
    probs_post = F.softplus(logits_post) + 1.0
    pred_cls_post = torch.argmax(probs_post / probs_post.sum(dim=1, keepdim=True), dim=1).squeeze().cpu().numpy()

    # Save Posterior Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(base_img, cmap='gray')
    axs[0].set_title(f"Original EM | Rank {rank}")
    axs[1].imshow(true_overlay)
    axs[1].set_title("Ground Truth Instance")
    axs[2].imshow(pred_cls_post, cmap='tab20', vmin=0, vmax=15)
    axs[2].set_title("Posterior Reconstruction (Upper Bound)")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(patch_dir, "01_posterior_reconstruction.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 2. Consensus Watershed & Epistemic Uncertainty (Standard)
    # ---------------------------------------------------------
    pred_instance = evidential_watershed_instance_map(net, img_t, device, n_samples=n_samples, variance_threshold=0.05)
    pred_instance = paper_appendix_h_post_processing(pred_instance, n=5, m=11)
    pred_overlay = label2rgb(pred_instance, image=base_img, bg_label=0, alpha=0.4)
    
    logits_det = net.sample(img_t, use_mean=True)
    evidence = F.softplus(logits_det)
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred_cls_det = torch.argmax(alpha / S, dim=1).squeeze().cpu().numpy()
    uncertainty_map = (NUM_CLASSES / S).squeeze().cpu().numpy()

    # Save Main Consensus Plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(base_img, cmap='gray')
    axs[0].set_title(f"Original EM | Score: {score:.4f}")
    axs[1].imshow(pred_cls_det, cmap='tab20', vmin=0, vmax=15)
    axs[1].set_title("Deterministic Semantic Prior")
    im_u = axs[2].imshow(uncertainty_map, cmap='magma')
    axs[2].set_title("Epistemic Uncertainty Map")
    plt.colorbar(im_u, ax=axs[2], fraction=0.046, pad=0.04)
    axs[3].imshow(pred_overlay)
    axs[3].set_title("Final Watershed Instances")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(patch_dir, "02_consensus_summary.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 3. Save the 16 Independent Monte Carlo Prior Samples
    # ---------------------------------------------------------
    q_prior = net.prior(img_t)['distributions']
    
    for i in range(n_samples):
        z_sample = [dist.rsample() for dist in q_prior]
        prior_dec = net.prior(encoder_features=enc_features, z_q=z_sample)['decoder_features']
        logits_sample = net.f_comb(enc_features, prior_dec)
        
        probs_sample = F.softplus(logits_sample) + 1.0
        pred_cls_sample = torch.argmax(probs_sample / probs_sample.sum(dim=1, keepdim=True), dim=1).squeeze().cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.imshow(pred_cls_sample, cmap='tab20', vmin=0, vmax=15)
        plt.title(f"Prior Stochastic Sample {i+1}/16")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"sample_{i+1:02d}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="/bhupender/2026/PUNET/new_punet/SNIMED/outputs_snemi3d/42_checkpoint_20260805_162559")
    parser.add_argument("--images_path", type=str, default="/bhupender/datasets/SNEMI3D/image/train-input.tif")
    parser.add_argument("--labels_path", type=str, default="/bhupender/datasets/SNEMI3D/seg/train-labels.tif")  
    parser.add_argument("--n_instance_samples", type=int, default=16)
    parser.add_argument("--max_instance_patches", type=int, default=None)
    parser.add_argument("--save_vis_dir", type=str, default="segmentation_overlays")
    args = parser.parse_args()

    os.makedirs(args.save_vis_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = HierarchicalEvidentialUNetSNEMI3D(
        input_channels=1, 
        num_classes=NUM_CLASSES,
        latent_dims=(40, 25, 15, 5),
        channels_per_block=(32, 64, 128, 256, 256, 256), 
        num_resblocks=2, 
    ).to(device)
    
    net = load_latest_checkpoint(net, args.checkpoint_dir, device=device)

    test_dataset = SNEMI3DPatches(args.images_path, args.labels_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    fg_metrics = evaluate_deterministic_performance(net, test_loader, device)
    print("\n--- Deterministic Inference (Prior) ---")
    for k, v in fg_metrics.items(): print(f"  {k}: {v:.4f}")

    print("\n--- Reconstruction Fidelity (Posterior) ---")
    try:
        posterior_iou = evaluate_posterior_reconstruction_iou(net, test_loader, device)
        print(f"  reconstruction_iou (NAN-mean): {posterior_iou:.4f}")
    except NotImplementedError as e:
        print(f"  [Error] {e}")

    uncertainty = evaluate_epistemic_uncertainty(net, test_loader, device)
    print(f"\nMean epistemic uncertainty (test set): {uncertainty:.4f}")

    print(f"\n--- Instance segmentation via {args.n_instance_samples}-sample Evidential Watershed Clustering ---")
    running = {"sum_comb_c": 0.0, "sum_comb_k": 0.0, "sum_comb_l": 0.0}
    n_scored = 0
    
    # Priority Queue restricted to 20 pointers. Prevents OOM memory leaks.
    top_20_patches_heap = []
    
    for idx, (z, y0, x0) in enumerate(test_dataset.anchors):
        if args.max_instance_patches is not None and idx >= args.max_instance_patches:
            break
            
        img_patch = test_dataset.images[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
        true_instance = test_dataset.labels[z, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] 

        img_t = torch.from_numpy(img_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        pred_instance = evidential_watershed_instance_map(
                net=net, patch=img_t, device=device, 
                n_samples=args.n_instance_samples, variance_threshold=0.05
        )

        pred_instance = paper_appendix_h_post_processing(pred_instance, n=5, m=11)

        # Global Accumulation
        accumulate_rand_terms(pred_instance, true_instance, running)
        
        # Local Accumulation for Top 20 Tracking
        local_running = {"sum_comb_c": 0.0, "sum_comb_k": 0.0, "sum_comb_l": 0.0}
        accumulate_rand_terms(pred_instance, true_instance, local_running)
        local_f_score = finalize_adapted_rand(local_running)["f_score"]

        # Push metadata pointer to Min-Heap.
        heapq.heappush(top_20_patches_heap, (local_f_score, idx))
        if len(top_20_patches_heap) > 20:
            heapq.heappop(top_20_patches_heap)

        n_scored += 1

    rand_metrics = finalize_adapted_rand(running)
    print(f"  Scored {n_scored} test patches")
    for k, v in rand_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---------------------------------------------------------
    # SECOND PASS: Generate Comprehensive Visualizations
    # ---------------------------------------------------------
    print(f"\n--- Generating Diagnostics for Top 20 Patches ---")
    
    # Sort descending (Rank 1 is the highest score)
    top_20_sorted = sorted(top_20_patches_heap, key=lambda x: x[0], reverse=True)
    
    for rank, (score, idx) in enumerate(top_20_sorted, start=1):
        print(f"  Processing Rank {rank:02d} | Patch Index {idx:04d} | F-Score {score:.4f}")
        
        render_comprehensive_patch_diagnostics(
            net=net, 
            test_dataset=test_dataset, 
            idx=idx, 
            rank=rank, 
            score=score, 
            save_base_dir=args.save_vis_dir, 
            device=device, 
            n_samples=args.n_instance_samples
        )
        
if __name__ == "__main__":
    main()
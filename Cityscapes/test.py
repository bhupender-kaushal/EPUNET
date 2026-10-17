import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.metrics import adapted_rand_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import your Cityscapes architecture and dataset
from load_cityscapes_data import CityscapesStaticDataset, NUM_CLASSES, IGNORE_INDEX
from hierarchical_model_cityscapes_new import HierarchicalEvidentialUNetCityscapes

# ==========================================
# 1. EVALUATION METRICS & VISUALIZATION
# ==========================================
def calculate_ap50(pred_mask, gt_mask, expected_p):
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids > 0] 
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids > 0]
    
    if len(pred_ids) == 0 and len(gt_ids) == 0: return 1.0
    if len(pred_ids) == 0 or len(gt_ids) == 0: return 0.0

    fg_prob = np.sum(expected_p[1:], axis=0) 
    
    scored_preds = []
    for p_id in pred_ids:
        p_mask = (pred_mask == p_id)
        confidence = fg_prob[p_mask].mean()
        scored_preds.append((confidence, p_id, p_mask))
        
    scored_preds.sort(key=lambda x: x[0], reverse=True)
    
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, (_, p_id, p_mask) in enumerate(scored_preds):
        for j, g_id in enumerate(gt_ids):
            g_mask = (gt_mask == g_id)
            intersection = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou_mat[i, j] = intersection / (union + 1e-8)
            
    tp, fp = 0, 0
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

def calculate_rand_and_iou(pred_mask, gt_mask):
    are, _, _ = adapted_rand_error(gt_mask, pred_mask)
    pred_fg = (pred_mask > 0)
    gt_fg = (gt_mask > 0)
    
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    iou_rec = intersection / (union + 1e-8) if union > 0 else 0.0
    
    return are, iou_rec

def save_visualizations(image_tensor, gt_mask, consensus_map, instance_map, batch_idx, save_dir):
    """Saves a 4-panel mathematical comparison of the distributional and topological spaces."""
    # Convert image tensor to [H, W, C] and normalize for visual plotting
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='prism')
    axes[1].set_title("Static Ground Truth Instances")
    axes[1].axis('off')
    
    axes[2].imshow(consensus_map, cmap='tab10')
    axes[2].set_title("Sampled Categorical Consensus")
    axes[2].axis('off')
    
    axes[3].imshow(instance_map, cmap='prism')
    axes[3].set_title("Evidential Topological Watershed")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"val_image_{batch_idx:04d}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 2. CORE INFERENCE ENGINE
# ==========================================
@torch.no_grad()
def evidential_cityscapes_inference(model, image, n_samples=16, variance_threshold=0.10):
    model.eval()
    mc_predictions = []
    mc_valid_masks = []
    mc_expected_p = []
    
    prior_out = model.prior(image)
    enc_cache = prior_out['encoder_features']
    
    for _ in range(n_samples):
        z_sample = [dist.rsample() for dist in prior_out['distributions']]
        prior_dec = model.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
        logits = model.f_comb(enc_cache, prior_dec)
        
        evidence = torch.clamp(F.softplus(logits), max=200.0)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        expected_p = alpha / S
        
        _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
        alpha_c = torch.gather(alpha, 1, pred_cls)
        var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
        valid_mask = (var_c < variance_threshold).float()
        
        mc_predictions.append(pred_cls)
        mc_valid_masks.append(valid_mask)
        mc_expected_p.append(expected_p.squeeze(0).cpu().numpy())

    mc_valid_masks = torch.cat(mc_valid_masks, dim=1) 
    
    avg_expected_p = np.mean(mc_expected_p, axis=0) # [C, H, W]
    consensus_pred = np.argmax(avg_expected_p, axis=0) # [H, W]
    
    final_valid_mask = (mc_valid_masks.mean(dim=1) > 0.5)[0].cpu().numpy() # [H, W]
    
    fg_prob = np.sum(avg_expected_p[1:], axis=0) 
    bg_mask = fg_prob < 0.5 
    
    confident_cores = consensus_pred.copy()
    confident_cores[~final_valid_mask] = IGNORE_INDEX
    confident_cores[bg_mask] = IGNORE_INDEX
    
    core_instances = np.zeros_like(confident_cores, dtype=np.int32)
    current_instance_id = 1
    
    for c in range(1, NUM_CLASSES):
        class_mask = (confident_cores == c)
        labeled_array, num_features = ndimage.label(class_mask)
        
        if num_features > 0:
            core_instances[class_mask] = labeled_array[class_mask] + current_instance_id - 1
            current_instance_id += num_features
            
    distance = ndimage.distance_transform_edt(core_instances > 0)
    final_instance_mask = watershed(-distance, core_instances, mask=(fg_prob > 0.3))
    
    # Returning consensus_pred to bridge the visualization
    # return final_instance_mask, avg_expected_p, consensus_pred
    return final_instance_mask, avg_expected_p, consensus_pred, core_instances

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing Cityscapes Validation Dataset...")
    val_dataset = CityscapesStaticDataset(root_dir="/bhupender/datasets/Cityscapes", mode="val") 
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    net = HierarchicalEvidentialUNetCityscapes(
        input_channels=3, 
        num_classes=NUM_CLASSES,
        latent_dims=(40, 25, 15, 5), 
        channels_per_block=(32, 64, 128, 256, 256, 256, 256, 256, 256), 
        num_resblocks=2 
    ).to(device)

    checkpoint_path = "./outputs_cityscapes/42_checkpoint_20260824_070412/model_best.pth" 
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    print(f"Loaded model_best.pth successfully from Epoch {checkpoint.get('epoch', 'Unknown')}. Commencing inference.")

    # Setup Visualization Directory
    vis_dir = "./outputs_cityscapes/eval_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    ious_50 = []
    # Dual-Tracking Arrays
    are_recon_list = []
    iou_recon_list = []
    are_sampled_list = []
    iou_sampled_list = []
    
    pbar = tqdm(val_loader, desc="Evaluating Topology")
    for batch_idx, data in enumerate(pbar):
        images, gt_mask = data[0].to(device), data[1].squeeze().numpy()
        
        # 1. Execute Pipeline (Now extracting core_instances)
        pred_instances, avg_expected_p, consensus_pred, core_instances = evidential_cityscapes_inference(
            model=net, 
            image=images, 
            n_samples=16, 
            variance_threshold=0.10
        )
        
        # 2. Compute Metrics on Reconstructed Topology (Watershed)
        ap50 = calculate_ap50(pred_instances, gt_mask, avg_expected_p)
        are_recon, iou_recon = calculate_rand_and_iou(pred_instances, gt_mask)
        
        # 3. Compute Metrics on Sampled Topology (Pre-Watershed)
        are_sampled, iou_sampled = calculate_rand_and_iou(core_instances, gt_mask)
        
        # Append Metrics
        ious_50.append(ap50)
        are_recon_list.append(are_recon)
        iou_recon_list.append(iou_recon)
        are_sampled_list.append(are_sampled)
        iou_sampled_list.append(iou_sampled)
        
        # Output Visualization
        save_visualizations(images[0], gt_mask, consensus_pred, pred_instances, batch_idx, vis_dir)
        
        pbar.set_postfix({
            'AP50': f'{np.mean(ious_50):.4f}', 
            'ARE_Rec': f'{np.mean(are_recon_list):.4f}',
            'ARE_Sam': f'{np.mean(are_sampled_list):.4f}'
        })

    print(f"\n========================================")
    print(f"Final Cityscapes Evidential Evaluation:")
    print(f"Mean AP50 (Reconstructed):   {np.mean(ious_50):.4f}")
    print(f"----------------------------------------")
    print(f"Mean Rand Error (Sampled):       {np.mean(are_sampled_list):.4f}")
    print(f"Mean Rand Error (Reconstructed): {np.mean(are_recon_list):.4f}")
    print(f"----------------------------------------")
    print(f"Mean Rec IoU (Sampled):          {np.mean(iou_sampled_list):.4f}")
    print(f"Mean Rec IoU (Reconstructed):    {np.mean(iou_recon_list):.4f}")
    print(f"========================================")
    print(f"Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()
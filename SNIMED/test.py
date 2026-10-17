import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from scipy import ndimage
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

# Import your model definition
from hierarchical_model_snemi3d import HierarchicalEvidentialUNetSNEMI3D
from load_snemi3d_data import NUM_CLASSES, IGNORE_INDEX

# ==========================================
# 1. CORE INFERENCE FUNCTIONS
# ==========================================
def greedy_hamming_clustering(masked_preds, threshold=8, ignore_index=255):
    """Groups pixels into unique cell instances based on their 16-sample signatures."""
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

@torch.no_grad()
def run_full_inference_pipeline(model, image, n_samples=16, variance_threshold=0.10, distance_threshold=8):
    """
    Math & Logic: 
    1. Sample the VAE N times to get permutation variations.
    2. Extract Dirichlet Variance to mask out ambiguous boundaries (pixels > variance_threshold become 255).
    3. Run Hamming Clustering strictly on confident core pixels.
    4. Apply Watershed to mathematically expand cores back into the 255 gaps.
    """
    model.eval()
    B, C, H, W = image.shape
    mc_predictions = []
    mc_valid_masks = []
    
    prior_out = model.prior(image)
    enc_cache = prior_out['encoder_features']
    
    # 1. Generate 16 Monte Carlo Samples
    for _ in range(n_samples):
        z_sample = [dist.rsample() for dist in prior_out['distributions']]
        prior_dec = model.prior(encoder_features=enc_cache, z_q=z_sample)['decoder_features']
        logits = model.f_comb(enc_cache, prior_dec)
        
        evidence = torch.clamp(F.softplus(logits), max=200.0)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        expected_p = alpha / S
        
        _, pred_cls = torch.max(expected_p, dim=1, keepdim=True)
        
        # Dirichlet Variance Extraction
        alpha_c = torch.gather(alpha, 1, pred_cls)
        var_c = (alpha_c * (S - alpha_c)) / ((S ** 2) * (S + 1.0))
        
        valid_mask = (var_c < variance_threshold).float()
        
        mc_predictions.append(pred_cls)
        mc_valid_masks.append(valid_mask)

    mc_predictions = torch.cat(mc_predictions, dim=1) 
    mc_valid_masks = torch.cat(mc_valid_masks, dim=1) 
    
    # 2. Consensus Variance Masking
    final_valid_mask = (mc_valid_masks.mean(dim=1, keepdim=True) > 0.5).long()
    masked_predictions = torch.where(final_valid_mask == 1, mc_predictions, torch.tensor(IGNORE_INDEX, device=image.device))
    
    # 3. Hamming Clustering (Operating on Batch Index 0)
    core_instances = greedy_hamming_clustering(masked_predictions[0], threshold=distance_threshold)
    core_instances = core_instances.cpu().numpy()
    
    # 4. Topological Watershed
    distance = ndimage.distance_transform_edt(core_instances > 0)
    final_instance_mask = watershed(-distance, core_instances, mask=(image[0, 0].cpu().numpy() > 0))
    
    return final_instance_mask

# ==========================================
# 2. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Network
    net = HierarchicalEvidentialUNetSNEMI3D(
        input_channels=1, 
        num_classes=NUM_CLASSES,
        latent_dims=(40, 25, 15, 5), 
        channels_per_block=(32, 64, 128, 256, 256, 256), 
        num_resblocks=2
    ).to(device)
    
    # Load Best Checkpoint
    checkpoint_path = "outputs_snemi3d/<YOUR_CHECKPOINT_DIR>/model_best.pth" # UPDATE THIS
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded model from Epoch {checkpoint['epoch']} with Val GED: {checkpoint['val_ged']:.5f}")
    
    # Load Test Image
    test_image_path = "/bhupender/datasets/SNEMI3D/image/train-input.tif" # Replace with test set path
    image_vol = tiff.imread(test_image_path)
    
    # Normalize (Standard SNEMI3D preprocessing)
    img_slice = image_vol[0].astype(np.float32)
    img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-8)
    
    # Convert to Tensor [1, 1, H, W]
    input_tensor = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run Pipeline
    print("Running 16-Sample Evidential Inference and Hamming Clustering...")
    instance_map = run_full_inference_pipeline(net, input_tensor)
    
    # Visualize Result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original EM")
    plt.imshow(img_slice, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Instance Segmentation")
    plt.imshow(instance_map, cmap='prism') # Prism maps integer IDs to distinct colors
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("final_segmentation.png", dpi=300)
    print("Saved final_segmentation.png")
import random
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.ndimage as nd

BACKGROUND_ID = 0
NUM_CELL_VARIANTS = 15
NUM_CLASSES = 1 + NUM_CELL_VARIANTS  # 16
IGNORE_INDEX = 255

PATCH_SIZE = 256
TRAIN_MARGIN = 32
TRAIN_WINDOW = PATCH_SIZE + 2 * TRAIN_MARGIN


def static_graph_coloring(instance_map, max_colors=15):
    """
    Deterministically maps instances to channels 1-15.
    Guarantees touching instances never share a color (solves boundary bleed),
    and guarantees the colors never change across epochs (solves Prior collapse).
    """
    out = np.zeros_like(instance_map)
    ignore_mask = (instance_map == IGNORE_INDEX)
    
    cell_mask = (instance_map > BACKGROUND_ID) & (~ignore_mask)
    instances = np.unique(instance_map[cell_mask])
    
    if len(instances) == 0:
        out[ignore_mask] = IGNORE_INDEX
        return out
        
    # 1. Build Adjacency Graph using fast shifted matrices
    temp_map = np.zeros_like(instance_map)
    temp_map[cell_mask] = instance_map[cell_mask]
    
    shifted_up = np.roll(temp_map, -1, axis=0)
    shifted_down = np.roll(temp_map, 1, axis=0)
    shifted_left = np.roll(temp_map, -1, axis=1)
    shifted_right = np.roll(temp_map, 1, axis=1)
    
    edges = set()
    for shifted in (shifted_up, shifted_down, shifted_left, shifted_right):
        mask = (temp_map > 0) & (shifted > 0) & (temp_map != shifted)
        pairs = np.unique(np.c_[temp_map[mask], shifted[mask]], axis=0)
        for u, v in pairs:
            edges.add((u, v))
            edges.add((v, u))
            
    adj_list = {i: set() for i in instances}
    for u, v in edges:
        adj_list[u].add(v)
        
    # 2. Deterministic Greedy Coloring (NO RANDOMNESS)
    # Sort instances to ensure consistent processing order
    instances = np.sort(instances) 
    
    color_assignment = {}
    available_colors = np.arange(1, max_colors + 1)
    
    for u in instances:
        neighbor_colors = {color_assignment[v] for v in adj_list[u] if v in color_assignment}
        valid_colors = np.setdiff1d(available_colors, list(neighbor_colors))
        
        if len(valid_colors) > 0:
            # Always pick the lowest available valid channel
            color_assignment[u] = valid_colors[0]
        else:
            # Fallback for crowded nodes: Pick channel 1 (rare on 2D slices)
            color_assignment[u] = available_colors[0]
            
    # 3. Apply the static mapping
    for u, c in color_assignment.items():
        out[(instance_map == u) & cell_mask] = c
        
    out[ignore_mask] = IGNORE_INDEX
    return out

def generate_synthetic_annotators(label_patch, num_annotators=4, max_shift=1):
    """Simulates inter-observer variance on razor-thin EM boundaries."""
    annotators = [label_patch] # Ground truth is Annotator 0
    ignore_mask = (label_patch == IGNORE_INDEX)
    
    for _ in range(num_annotators - 1):
        bg_mask = (label_patch == BACKGROUND_ID) & (~ignore_mask)
        if random.random() > 0.5:
            shifted_bg = nd.binary_dilation(bg_mask, iterations=max_shift)
            new_ann = label_patch.copy()
            new_ann[shifted_bg] = BACKGROUND_ID
        else:
            shifted_bg = nd.binary_erosion(bg_mask, iterations=max_shift)
            new_ann = label_patch.copy()
            
        annotators.append(new_ann)
        
    return np.stack(annotators, axis=0) 

class SNEMI3DPatches(Dataset):
    def __init__(self, images_path, labels_path, mode="train", seed=None):
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.rng = random.Random(seed) if seed is not None else random

        images = tiff.imread(images_path)
        labels = tiff.imread(labels_path)

        z_ranges = {"train": (0, 80), "val": (80, 90), "test": (90, 100)}
        z0, z1 = z_ranges[mode]
        self.images = images[z0:z1]
        self.labels = labels[z0:z1]

        n_z, H, W = self.images.shape
        self.grid_h = H // PATCH_SIZE
        self.grid_w = W // PATCH_SIZE

        self.anchors = [(z, gy * PATCH_SIZE, gx * PATCH_SIZE)
                        for z in range(n_z) for gy in range(self.grid_h) for gx in range(self.grid_w)]

    def __len__(self): return len(self.anchors)

    def _extract_train_window(self, z, y0, x0):
        H, W = self.images.shape[1:]
        cy, cx = y0 + PATCH_SIZE // 2, x0 + PATCH_SIZE // 2
        half = TRAIN_WINDOW // 2

        img_canvas = np.zeros((TRAIN_WINDOW, TRAIN_WINDOW), dtype=self.images.dtype)
        lbl_canvas = np.full((TRAIN_WINDOW, TRAIN_WINDOW), IGNORE_INDEX, dtype=np.int64)

        sy0, sy1 = max(0, cy - half), min(H, cy + half)
        sx0, sx1 = max(0, cx - half), min(W, cx + half)
        dy0, dx0 = sy0 - (cy - half), sx0 - (cx - half)
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

        img_canvas[dy0:dy1, dx0:dx1] = self.images[z, sy0:sy1, sx0:sx1]
        lbl_canvas[dy0:dy1, dx0:dx1] = self.labels[z, sy0:sy1, sx0:sx1]
        return img_canvas, lbl_canvas

    def _augment(self, img_t, lbl_t):
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        angle = self.rng.uniform(-5, 5) 
        scale = self.rng.uniform(0.95, 1.05)
        
        img_t = TF.affine(img_t, angle=angle, translate=[0, 0], scale=scale, shear=[0],
                           interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        lbl_t = TF.affine(lbl_t.float(), angle=angle, translate=[0, 0], scale=scale, shear=[0],interpolation=InterpolationMode.NEAREST, fill=[float(IGNORE_INDEX)]).long()

        if self.rng.random() < 0.5: img_t, lbl_t = TF.hflip(img_t), TF.hflip(lbl_t)
        if self.rng.random() < 0.5: img_t, lbl_t = TF.vflip(img_t), TF.vflip(lbl_t)

        _, H, W = img_t.shape
        x1 = self.rng.randint(0, W - PATCH_SIZE)
        y1 = self.rng.randint(0, H - PATCH_SIZE)
        
        return img_t[:, y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE], lbl_t[:, y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE]

    def __getitem__(self, idx):
        z, y0, x0 = self.anchors[idx]

        if self.mode == "train":
            img_raw, lbl_raw = self._extract_train_window(z, y0, x0)
        else:
            img_raw = self.images[z, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            lbl_raw = self.labels[z, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE].astype(np.int64)

        img_t = torch.from_numpy(img_raw.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_t = torch.from_numpy(lbl_raw).long().unsqueeze(0)

        if self.mode == "train": 
            img_t, lbl_t = self._augment(img_t, lbl_t)

        lbl_np = lbl_t.squeeze(0).numpy()
        
        # Apply the graph coloring algorithm dynamically
        lbl_colored = static_graph_coloring(lbl_np, max_colors=NUM_CELL_VARIANTS)
        
        # Generate the multi-annotator simulations using the newly colored mask
        multi_masks = generate_synthetic_annotators(lbl_colored, num_annotators=4)
        
        return (img_t, 
                torch.from_numpy(lbl_colored).long().unsqueeze(0), 
                torch.from_numpy(multi_masks).long(), 
                {"z": z, "y0": y0, "x0": x0})
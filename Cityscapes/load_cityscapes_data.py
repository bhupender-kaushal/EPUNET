import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Background = 0, Car IDs = 1 through 5
NUM_CELL_VARIANTS = 5 
NUM_CLASSES = 1 + NUM_CELL_VARIANTS  # 6
IGNORE_INDEX = 255
PATCH_SIZE = 256 # Can be scaled up to 512 depending on your GPU VRAM

# Cityscapes 19 Evaluation Classes (Semantic IDs)
# Anything NOT in this list is mapped to IGNORE_INDEX (255)
VALID_SEMANTIC_IDS = {
    7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33
}
CAR_SEMANTIC_ID = 26

def static_cityscapes_coloring_and_simulation(instance_map, max_colors=5, num_annotators=4):
    """
    1. Deterministically colors instances based on physical adjacency.
    2. Generates synthetic multi-annotator boundary variance.
    """
    out = np.zeros_like(instance_map, dtype=np.int64)
    semantic_map = instance_map // 1000
    
    # Isolate valid pixels
    ignore_mask = np.ones_like(semantic_map, dtype=bool)
    for valid_id in VALID_SEMANTIC_IDS: # (Assume VALID_SEMANTIC_IDS is defined)
        ignore_mask &= (semantic_map != valid_id)
        
    car_mask = (semantic_map == CAR_SEMANTIC_ID)
    car_instances = np.unique(instance_map[car_mask])
    
    # 1. Deterministic Coloring (Sorted instance IDs guarantee static assignment)
    if len(car_instances) > 0:
        car_instances = np.sort(car_instances)
        for i, car_id in enumerate(car_instances):
            color = (i % max_colors) + 1
            out[instance_map == car_id] = color
            
    out[ignore_mask] = IGNORE_INDEX
    
    # 2. Synthetic Inter-Observer Variance
    annotators = [out]
    for _ in range(num_annotators - 1):
        bg_mask = (out == 0) & (~ignore_mask)
        new_ann = out.copy()
        if random.random() > 0.5:
            shifted_bg = ndimage.binary_dilation(bg_mask, iterations=1)
            new_ann[shifted_bg] = 0
        else:
            shifted_bg = ndimage.binary_erosion(bg_mask, iterations=1)
        annotators.append(new_ann)
        
    # Returns [4, H, W] tensor of static but boundary-ambiguous masks
    return np.stack(annotators, axis=0)

def dynamic_ambiguous_coloring(instance_map, max_colors=5):
    """
    Executes the Kohl et al. (2018) Ambiguous Coloring trick for Cityscapes.
    Randomly assigns a channel (1-5) to distinct cars every time the image is loaded.
    """
    out = np.zeros_like(instance_map, dtype=np.int64)
    
    # 1. Isolate semantic base IDs (Instance IDs are semantic_id * 1000 + instance_count)
    semantic_map = instance_map // 1000
    
    # 2. Build the Ignore Mask (Pixels not belonging to the 19 valid classes)
    ignore_mask = np.ones_like(semantic_map, dtype=bool)
    for valid_id in VALID_SEMANTIC_IDS:
        ignore_mask &= (semantic_map != valid_id)
        
    # 3. Find unique car instances
    # Car instances are IDs >= 26000 and < 27000. 
    # (ID exactly 26000 is usually a crowd/group of cars, but we treat it as valid per base rules unless explicitly excluded)
    car_mask = (semantic_map == CAR_SEMANTIC_ID)
    car_instances = np.unique(instance_map[car_mask])
    
    # 4. The Coloring Trick
    if len(car_instances) > 0:
        available_colors = list(range(1, max_colors + 1))
        random.shuffle(available_colors)
        
        for i, car_id in enumerate(car_instances):
            # If > 5 cars, it wraps around (collision ambiguity)
            color = available_colors[i % len(available_colors)]
            out[instance_map == car_id] = color
            
    # Apply ignore mask last to overwrite any overlapping padding
    out[ignore_mask] = IGNORE_INDEX
    
    return out

class CityscapesStaticDataset(Dataset):
# class CityscapesColoringDataset(Dataset):
    def __init__(self, root_dir, mode="train", seed=None):
        """
        root_dir: path to cityscapes directory containing 'leftImg8bit' and 'gtFine'
        """
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.rng = random.Random(seed) if seed is not None else random
        self.root_dir = root_dir
        
        # Load file paths
        search_path = os.path.join(root_dir, 'leftImg8bit', mode, '*', '*_leftImg8bit.png')
        self.images = sorted(glob.glob(search_path))
        
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found at {search_path}")

    def __len__(self): 
        return len(self.images)
        


    def _get_target_path(self, img_path):
            # Isolate directory and filename
            dir_name, file_name = os.path.split(img_path)
            
            # Replace the directory path
            new_dir = dir_name.replace('leftImg8bit', 'gtFine')
            
            # Replace the filename suffix
            new_file = file_name.replace('_leftImg8bit.png', '_gtFine_instanceIds.png')
            
            return os.path.join(new_dir, new_file)

    def _augment(self, img_t, lbl_t):
        """Applies spatial and color deformations while safely masking padded areas."""
        # 1. Color Jitter (Images only)
        if self.rng.random() < 0.5:
            brightness = self.rng.uniform(0.8, 1.2)
            contrast = self.rng.uniform(0.8, 1.2)
            img_t = TF.adjust_brightness(img_t, brightness)
            img_t = TF.adjust_contrast(img_t, contrast)

        # 2. Affine Deformations (Angle, Scale, Shear)
        angle = self.rng.uniform(-10, 10) 
        scale = self.rng.uniform(0.9, 1.1)
        shear = self.rng.uniform(-5, 5)
        
        # The padding generated by the affine transformation on the label tensor is marked as IGNORE_INDEX
        img_t = TF.affine(img_t, angle=angle, translate=[0, 0], scale=scale, shear=[shear],
                           interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        lbl_t = TF.affine(lbl_t.float(), angle=angle, translate=[0, 0], scale=scale, shear=[shear],
                           interpolation=InterpolationMode.NEAREST, fill=[float(IGNORE_INDEX)]).long()

        if self.rng.random() < 0.5: 
            img_t, lbl_t = TF.hflip(img_t), TF.hflip(lbl_t)

        # 3. Random Translated Crop
        _, H, W = img_t.shape
        if H > PATCH_SIZE and W > PATCH_SIZE:
            y1 = self.rng.randint(0, H - PATCH_SIZE)
            x1 = self.rng.randint(0, W - PATCH_SIZE)
            img_t = img_t[:, y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE]
            lbl_t = lbl_t[:, y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE]
        else:
            img_t = TF.center_crop(img_t, PATCH_SIZE)
            lbl_t = TF.center_crop(lbl_t, PATCH_SIZE)
            
        return img_t, lbl_t

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target_path = self._get_target_path(img_path)
        
        # Load Images
        image = Image.open(img_path).convert('RGB')
        target = Image.open(target_path) # Mode 'I' (16-bit integer)
        
        # 1. Paper Requirement: Downsample to half-resolution (512 x 1024)
        image = TF.resize(image, (512, 1024), interpolation=InterpolationMode.BILINEAR)
        target = TF.resize(target, (512, 1024), interpolation=InterpolationMode.NEAREST)
        
        img_t = TF.to_tensor(image) # [3, H, W]
        lbl_t = torch.from_numpy(np.array(target, dtype=np.int32)).unsqueeze(0) # [1, H, W]

        # 2. Augmentations (Train only)
        if self.mode == "train": 
            img_t, lbl_t = self._augment(img_t, lbl_t)
        else:
            img_t = TF.center_crop(img_t, PATCH_SIZE)
            lbl_t = TF.center_crop(lbl_t, PATCH_SIZE)

        lbl_np = lbl_t.squeeze(0).numpy()
        
        # 3. Apply the Ambiguous Graph Coloring Trick
        lbl_colored = dynamic_ambiguous_coloring(lbl_np, max_colors=NUM_CELL_VARIANTS)
        
        # 4. Interface Alignment
        # Tile the colored mask 4 times to satisfy the SNEMI3D `multi_mask_labels` tensor shape
        # Since Cityscapes doesn't have multiple annotators, we use exact duplicates.
        # The Evidential MC loop will just sample the same target 4 times (statistically identical).
        multi_masks = np.stack([lbl_colored] * 4, axis=0)
        
        return (img_t, 
                torch.from_numpy(lbl_colored).long().unsqueeze(0), 
                torch.from_numpy(multi_masks).long(), 
                {"path": img_path}) # Passed path instead of coordinates
                
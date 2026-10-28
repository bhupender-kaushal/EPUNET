import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import pickle
import os
from glob import glob
import matplotlib.image  # explicit: matplotlib.image is not loaded by "import matplotlib"


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        # label = self.labels[index].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

class LIDC_IDRI_Folder(Dataset):
    def __init__(self, dataset_location, mode='train', transform=None, allow_missing_gt=False):
        self.dataset_location = dataset_location
        self.mode = mode.lower()
        self.transform = transform

        # 1. Find all image files across ALL patient folders
        # This makes the dataset recognize every individual slice
        self.all_image_paths = sorted(glob(os.path.join(dataset_location, 'images', '*', '*.png')))

        if not self.all_image_paths:
            raise RuntimeError(f"No .png images found in {dataset_location}/images")

        # F6: a wrong path would silently turn every label into an empty mask
        num_missing = sum(
            1 for p in self.all_image_paths
            if not all(os.path.exists(p.replace('images', 'gt').replace('.png', f'_l{l}.png')) for l in range(4))
        )
        print(f"[*] {dataset_location}: {len(self.all_image_paths)} images, {num_missing} with missing GT files")
        if num_missing > 0 and not allow_missing_gt:
            raise RuntimeError(f"{num_missing} images in {dataset_location} have missing gt/..._l{{0..3}}.png files "
                               f"(pass allow_missing_gt=True to fill them with empty masks)")

    def __len__(self):
        # Now returns the total count of images/slices
        return len(self.all_image_paths)

    def __getitem__(self, index):
        # 2. Get the specific image path for this index
        img_path = self.all_image_paths[index]
        
        # Extract the patient ID (the folder name) for the series_uid
        # path is: .../images/patient_id/image_name.png
        
        
        # series_uid = os.path.basename(os.path.dirname(img_path))
        series_uid = img_path


        # 3. Load Image
        image = matplotlib.image.imread(img_path).astype(np.float32)
        
        # 4. Load Masks logic (same as before)
        gt_base_path = img_path.replace('images', 'gt')
        
        if self.mode == 'train':
            l_idx = random.randint(0, 3)
            gt_path = gt_base_path.replace('.png', f'_l{l_idx}.png')
            label = matplotlib.image.imread(gt_path).astype(np.float32) if os.path.exists(gt_path) else np.zeros_like(image)
            label = np.expand_dims(label, axis=0)
        else:
            labels = []
            for l in range(4):
                gt_path = gt_base_path.replace('.png', f'_l{l}.png')
                labels.append(matplotlib.image.imread(gt_path).astype(np.float32) if os.path.exists(gt_path) else np.zeros_like(image))
            label = np.stack(labels, axis=0)

        # 5. Crop & Process
        image = image[26:-26, 26:-26]
        label = label[..., 26:-26, 26:-26]
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image), torch.from_numpy(label), series_uid


class LIDC_IDRI_test(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        # label = self.labels[index][random.randint(0,3)].astype(float)

        label = np.array(self.labels[index]).astype(float)
        if self.transform is not None:
            image = self.transform(image)

        series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label) 

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
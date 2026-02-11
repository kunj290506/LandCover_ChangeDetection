import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    """Memory-optimized change detection dataset"""
    
    def __init__(self, data_root: str, list_file: str, mode: str = 'train', patch_size: int = 256):
        self.data_root = data_root
        self.mode = mode
        self.patch_size = patch_size
        
        # Load file paths
        self.samples = []
        list_path = os.path.join(data_root, list_file)
        if os.path.exists(list_path):
            with open(list_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.samples.append({
                            'image1': os.path.join(data_root, parts[0]),
                            'image2': os.path.join(data_root, parts[1]),
                            'label': os.path.join(data_root, parts[2])
                        })
        
        # Lightweight augmentations for memory efficiency
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(patch_size, patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'image2': 'image', 'mask': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(patch_size, patch_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'image2': 'image', 'mask': 'mask'})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images with memory optimization
        try:
            image1 = cv2.imread(sample['image1'])
            if image1 is None:
                raise FileNotFoundError(f"Could not load {sample['image1']}")
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            
            image2 = cv2.imread(sample['image2'])
            if image2 is None:
                raise FileNotFoundError(f"Could not load {sample['image2']}")
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not load {sample['label']}")
            mask = (mask > 127).astype(np.uint8)
            
            # Apply transforms
            transformed = self.transform(image=image1, image2=image2, mask=mask)
            
            return {
                'image1': transformed['image'],
                'image2': transformed['image2'], 
                'label': transformed['mask'].unsqueeze(0).float()
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data if file loading fails
            dummy_img = torch.zeros(3, self.patch_size, self.patch_size)
            dummy_mask = torch.zeros(1, self.patch_size, self.patch_size)
            return {
                'image1': dummy_img,
                'image2': dummy_img,
                'label': dummy_mask
            }
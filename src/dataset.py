import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from PIL import Image

import torchvision.transforms as transforms

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, list_path, mode='train', patch_size=256, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            list_path (string): Path to the text file containing the list of image pairs.
                                Format: image_A_path image_B_path label_path
            mode (string): 'train', 'val', or 'test'.
            patch_size (int): Size of patches to extract/resize to.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        self.transform = transform
        self.list_path = list_path
        
        self.files = []
        with open(list_path, 'r') as f:
            for line in f:
                self.files.append(line.strip().split())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Paths are relative to root_dir
        img1_path = os.path.join(self.root_dir, self.files[idx][0])
        img2_path = os.path.join(self.root_dir, self.files[idx][1])
        label_path = os.path.join(self.root_dir, self.files[idx][2])

        # Open images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path).convert('L') # Binary mask

        # Augmentation
        if self.mode == 'train':
            img1, img2, label = self._transform(img1, img2, label)
        else:
            # Just resize or center crop if needed, or pass as is if already preprocessed
            pass
            
        # To Tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        # For label: convert to tensor and ensure binary (0 or 1)
        # TF.to_tensor scales by 255, so 0->0, 255->1, 1->0.004
        label = TF.to_tensor(label)
        # If original labels were 0/1 (not 0/255), they become 0/0.004
        # Check and fix: if max < 0.1, the original was 0/1, so threshold at 0.001
        if label.max() > 0 and label.max() < 0.1:
            label = (label > 0.001).float()
        elif label.max() > 0.1:
            # Original was 0/255, now 0/1, threshold at 0.5
            label = (label > 0.5).float()

        # Normalize (custom or ImageNet statistics)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img1 = TF.normalize(img1, mean, std)
        img2 = TF.normalize(img2, mean, std)

        return {'image1': img1, 'image2': img2, 'label': label, 'name': self.files[idx][0]}

    def _transform(self, img1, img2, label):
        # Random Crop
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(self.patch_size, self.patch_size))
            img1 = TF.crop(img1, i, j, h, w)
            img2 = TF.crop(img2, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
        
        # Random Horizontal Flip
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)

        # Random Vertical Flip
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)
            
        # Random Rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            label = TF.rotate(label, angle)

        # Random Color Jitter (Brightness, Contrast, Saturation, Hue)
        if random.random() > 0.5:
            # Apply samejitter to both images to simulate same sensor conditions, 
            # or different to simulate different conditions?
            # For change detection, we ideally want robust to different conditions.
            # Applying independently or same? 
            # Let's apply slightly different jitters or same with probability.
            
            # Define jitter transform
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            
            # Apply to both
            img1 = jitter(img1)
            img2 = jitter(img2)

        # Random Gaussian Blur
        if random.random() > 0.5:
            sigma = random.uniform(0.1, 2.0)
            img1 = TF.gaussian_blur(img1, kernel_size=3, sigma=sigma)
            img2 = TF.gaussian_blur(img2, kernel_size=3, sigma=sigma)

        return img1, img2, label

# Preprocessing utils (e.g. for patch extraction from large files)
def extract_patches(image, patch_size=256, stride=256):
    # This would be a utility to pre-process large satellite scenes into chips
    # Implementation placeholder
    pass

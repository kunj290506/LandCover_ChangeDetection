"""
Land Cover Change Detection - Model Inference Module
Handles model loading, preprocessing, and prediction
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================
# Model Architecture (SNUNet with CBAM)
# ============================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.sa(x * self.ca(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch) if use_cbam else None

    def forward(self, x):
        x = self.conv(x)
        return self.cbam(x) if self.cbam else x


class SNUNet(nn.Module):
    """Siamese Nested U-Net with CBAM Attention"""
    
    def __init__(self, in_ch=3, num_classes=1, C=32, use_attn=True):
        super().__init__()
        # Encoder
        self.conv0_0 = ConvBlock(in_ch, C)
        self.pool1 = nn.MaxPool2d(2)
        self.conv1_0 = ConvBlock(C, C*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(C*2, C*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(C*4, C*8)
        self.pool4 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(C*8, C*16)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_1 = ConvBlock(C*2 + C*4, C, use_cbam=use_attn)
        self.conv1_1 = ConvBlock(C*4 + C*8, C*2, use_cbam=use_attn)
        self.conv2_1 = ConvBlock(C*8 + C*16, C*4, use_cbam=use_attn)
        self.conv3_1 = ConvBlock(C*16 + C*32, C*8, use_cbam=use_attn)
        self.conv0_2 = ConvBlock(C*2 + C*2 + C, C, use_cbam=use_attn)
        self.conv1_2 = ConvBlock(C*4 + C*4 + C*2, C*2, use_cbam=use_attn)
        self.conv2_2 = ConvBlock(C*8 + C*8 + C*4, C*4, use_cbam=use_attn)
        self.conv0_3 = ConvBlock(C*2 + C*2 + C + C, C, use_cbam=use_attn)
        self.conv1_3 = ConvBlock(C*4 + C*4 + C*2 + C*2, C*2, use_cbam=use_attn)
        self.conv0_4 = ConvBlock(C*2 + C*2 + C + C + C, C)
        self.final = nn.Conv2d(C, num_classes, 1)
        
    def forward(self, x1, x2):
        # Encoder 1
        x1_0_0 = self.conv0_0(x1)
        x1_1_0 = self.conv1_0(self.pool1(x1_0_0))
        x1_2_0 = self.conv2_0(self.pool2(x1_1_0))
        x1_3_0 = self.conv3_0(self.pool3(x1_2_0))
        x1_4_0 = self.conv4_0(self.pool4(x1_3_0))
        # Encoder 2
        x2_0_0 = self.conv0_0(x2)
        x2_1_0 = self.conv1_0(self.pool1(x2_0_0))
        x2_2_0 = self.conv2_0(self.pool2(x2_1_0))
        x2_3_0 = self.conv3_0(self.pool3(x2_2_0))
        x2_4_0 = self.conv4_0(self.pool4(x2_3_0))
        # Decoder
        x0_1 = self.conv0_1(torch.cat([x1_0_0, x2_0_0, self.up(x1_1_0), self.up(x2_1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_1_0, x2_1_0, self.up(x1_2_0), self.up(x2_2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x1_2_0, x2_2_0, self.up(x1_3_0), self.up(x2_3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x1_3_0, x2_3_0, self.up(x1_4_0), self.up(x2_4_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x1_0_0, x2_0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_1_0, x2_1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x1_2_0, x2_2_0, x2_1, self.up(x3_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x1_0_0, x2_0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_1_0, x2_1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x1_0_0, x2_0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(x0_4)


# ============================================
# Change Detector Class
# ============================================

class ChangeDetector:
    """High-level interface for change detection inference."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the change detector.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Model path not found: {model_path}")
            print("Running in demo mode (random predictions)")
    
    def load_model(self, model_path):
        """Load model from checkpoint."""
        try:
            self.model = SNUNet(in_ch=3, num_classes=1, C=32, use_attn=True)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_loaded = False
    
    def preprocess(self, image_path):
        """
        Preprocess image for model input.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize to standard size if needed
        target_size = 256
        if img.size[0] != target_size or img.size[1] != target_size:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, self.mean, self.std)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image1_path, image2_path, threshold=0.5):
        """
        Run change detection on image pair.
        
        Args:
            image1_path: Path to before image
            image2_path: Path to after image
            threshold: Binary threshold for change mask
            
        Returns:
            Dictionary with mask, overlay, and metrics
        """
        # Load and preprocess images
        img1 = self.preprocess(image1_path).to(self.device)
        img2 = self.preprocess(image2_path).to(self.device)
        
        if self.model_loaded and self.model is not None:
            # Run model inference
            with torch.no_grad():
                output = self.model(img1, img2)
                prob = torch.sigmoid(output)
                mask = (prob > threshold).float()
        else:
            # Demo mode: generate random mask
            h, w = img1.shape[2], img1.shape[3]
            mask = torch.zeros(1, 1, h, w)
            # Add some random "changes"
            cx, cy = np.random.randint(h//4, 3*h//4), np.random.randint(w//4, 3*w//4)
            r = np.random.randint(20, 50)
            y, x = np.ogrid[:h, :w]
            circle = ((x - cx)**2 + (y - cy)**2 <= r**2).astype(np.float32)
            mask[0, 0] = torch.from_numpy(circle)
            prob = mask
        
        # Convert to numpy
        mask_np = mask[0, 0].cpu().numpy()
        prob_np = prob[0, 0].cpu().numpy()
        
        # Create mask image (white = change)
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        
        # Create overlay visualization
        img2_pil = Image.open(image2_path).convert('RGB')
        if img2_pil.size != (256, 256):
            img2_pil = img2_pil.resize((256, 256), Image.BILINEAR)
        
        overlay = self._create_overlay(img2_pil, mask_np)
        
        # Calculate metrics
        changed_pixels = int(mask_np.sum())
        confidence = float(prob_np.mean()) if changed_pixels > 0 else 0.0
        
        return {
            'mask': mask_img,
            'overlay': overlay,
            'changed_pixels': changed_pixels,
            'confidence': confidence,
            'probability_map': prob_np
        }
    
    def _create_overlay(self, image, mask, color=(255, 0, 100), alpha=0.5):
        """
        Create overlay visualization with change mask.
        
        Args:
            image: PIL Image (background)
            mask: numpy array (binary mask)
            color: RGB tuple for overlay color
            alpha: Transparency of overlay
            
        Returns:
            PIL Image with overlay
        """
        img_array = np.array(image).astype(np.float32)
        
        # Create colored overlay
        overlay = np.zeros_like(img_array)
        overlay[mask > 0] = color
        
        # Blend
        result = img_array.copy()
        blend_mask = mask[:, :, np.newaxis] > 0
        result = np.where(
            blend_mask,
            img_array * (1 - alpha) + overlay * alpha,
            img_array
        )
        
        return Image.fromarray(result.astype(np.uint8))


# ============================================
# CLI Interface
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run change detection inference')
    parser.add_argument('--image1', required=True, help='Path to before image')
    parser.add_argument('--image2', required=True, help='Path to after image')
    parser.add_argument('--model', default='best_model.pth', help='Path to model')
    parser.add_argument('--output', default='result_mask.png', help='Output path')
    
    args = parser.parse_args()
    
    detector = ChangeDetector(args.model)
    result = detector.predict(args.image1, args.image2)
    
    result['mask'].save(args.output)
    print(f"Saved mask to {args.output}")
    print(f"Changed pixels: {result['changed_pixels']}")

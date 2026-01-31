import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.snunet import SNUNet
import torch.nn.functional as F

# Try importing rasterio
try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("Rasterio not installed. GeoTIFF support will be limited/unavailable.")
    rasterio = None

def load_model(model_path, device):
    model = SNUNet(3, 1, use_attention=True)
    # Check if pruned/quantized or standard
    checkpoint = torch.load(model_path, map_location=device)
    # Handle state dict if it was saved as clean state dict or full checkpoint
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def predict_sliding_window(model, img1_path, img2_path, output_path, patch_size=256, stride=256, device='cuda'):
    if rasterio is None:
        print("Cannot process GeoTIFFs without rasterio.")
        return

    with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
        profile = src1.profile
        width = src1.width
        height = src1.height
        
        # Update profile for output (single channel, uint8)
        profile.update(count=1, dtype=rasterio.uint8)
        
        # Prepare output array
        # For very large images, write directly to file block by block, but here assuming memory fits or using Memfile
        # Better: Process and write continuously.
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for y in tqdm(range(0, height, stride)):
                for x in range(0, width, stride):
                    # Define window
                    window = Window(x, y, min(patch_size, width - x), min(patch_size, height - y))
                    
                    # Read data
                    img1_chip = src1.read(window=window) # (C, H, W)
                    img2_chip = src2.read(window=window)
                    
                    # Handle edge cases where chip is smaller than patch_size
                    # Pad if necessary
                    c, h, w = img1_chip.shape
                    if h < patch_size or w < patch_size:
                        # Simple padding: replicate or constant
                        pad_h = patch_size - h
                        pad_w = patch_size - w
                        img1_chip = np.pad(img1_chip, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                        img2_chip = np.pad(img2_chip, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
                        
                    # Preprocess (normalize)
                    img1_tensor = torch.from_numpy(img1_chip).float().unsqueeze(0) / 255.0 # (1, C, H, W)
                    img2_tensor = torch.from_numpy(img2_chip).float().unsqueeze(0) / 255.0
                    
                    # Normalize using ImageNet stats if used in training
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    img1_tensor = (img1_tensor - mean) / std
                    img2_tensor = (img2_tensor - mean) / std
                    
                    img1_tensor = img1_tensor.to(device)
                    img2_tensor = img2_tensor.to(device)
                    
                    with torch.no_grad():
                        output = model(img1_tensor, img2_tensor)
                        prob = torch.sigmoid(output)
                        pred = (prob > 0.5).float().cpu().numpy().squeeze() # (H, W)
                        
                    # Crop back if padded
                    if h < patch_size or w < patch_size:
                        pred = pred[:h, :w]
                        
                    # Write to file
                    # Ensure shape is (1, H, W)
                    dst.write(pred.astype(rasterio.uint8) * 255, 1, window=window)

    print(f"Inference complete. Saved to {output_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.model_path, device)
    
    if os.path.isdir(args.input_dir):
        # Batch processing mode (not implemented fully, placeholder)
        print("Processing directory...")
        pass
    else:
        # Single pair processing
        predict_sliding_window(model, args.img1, args.img2, args.output, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img1', type=str, help='Path to pre-change image')
    parser.add_argument('--img2', type=str, help='Path to post-change image')
    parser.add_argument('--output', type=str, default='output.tif', help='Output path')
    parser.add_argument('--input_dir', type=str, help='Optional: directory for batch processing')
    
    args = parser.parse_args()
    main(args)

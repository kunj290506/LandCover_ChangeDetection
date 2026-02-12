#!/usr/bin/env python3
"""
Download Unseen Test Images for Land Cover Change Detection
Downloads sample satellite images for testing the trained model
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

def create_test_directory():
    """Create directory for test images"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (test_dir / "before").mkdir(exist_ok=True)
    (test_dir / "after").mkdir(exist_ok=True)
    (test_dir / "results").mkdir(exist_ok=True)
    
    return test_dir

def download_file(url, filepath):
    """Download file from URL"""
    try:
        print(f"📥 Downloading: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def create_synthetic_test_images(test_dir):
    """Create synthetic test images for demonstration"""
    print("🎨 Creating synthetic test images...")
    
    # Create base landscape image
    def create_landscape(size=(512, 512), seed=42):
        np.random.seed(seed)
        
        # Create base terrain
        h, w = size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add terrain layers
        # Green areas (forests/vegetation)
        for _ in range(20):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(20, 80)
            cv2.circle(image, (x, y), radius, (34, 139, 34), -1)
        
        # Water bodies (blue)
        for _ in range(5):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(30, 60)
            cv2.circle(image, (x, y), radius, (30, 144, 255), -1)
        
        # Urban areas (gray)
        for _ in range(8):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            w_rect, h_rect = np.random.randint(40, 100), np.random.randint(40, 100)
            cv2.rectangle(image, (x, y), (x+w_rect, y+h_rect), (128, 128, 128), -1)
        
        # Add some noise and smoothing
        noise = np.random.randint(0, 30, (h, w, 3))
        image = cv2.add(image, noise.astype(np.uint8))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
        return image
    
    # Create "before" image
    before_image = create_landscape(seed=42)
    
    # Create "after" image with changes
    after_image = before_image.copy()
    
    # Add deforestation (remove green, add brown)
    cv2.rectangle(after_image, (100, 100), (200, 200), (139, 69, 19), -1)
    cv2.rectangle(after_image, (300, 300), (400, 400), (139, 69, 19), -1)
    
    # Add new urban development
    cv2.rectangle(after_image, (150, 350), (250, 450), (105, 105, 105), -1)
    cv2.rectangle(after_image, (50, 450), (120, 500), (105, 105, 105), -1)
    
    # Add water body change
    cv2.circle(after_image, (400, 100), 40, (30, 144, 255), -1)
    
    # Resize to 256x256 for model compatibility
    before_image = cv2.resize(before_image, (256, 256))
    after_image = cv2.resize(after_image, (256, 256))
    
    # Save images
    before_path = test_dir / "before" / "synthetic_before.png"
    after_path = test_dir / "after" / "synthetic_after.png"
    
    cv2.imwrite(str(before_path), before_image)
    cv2.imwrite(str(after_path), after_image)
    
    print(f"✅ Created synthetic test images:")
    print(f"   Before: {before_path}")
    print(f"   After: {after_path}")
    
    return before_path, after_path

def create_real_world_samples(test_dir):
    """Create more realistic test samples"""
    print("🌍 Creating realistic test samples...")
    
    # Sample 1: Forest to urban conversion
    def create_forest_urban_change():
        # Before: Dense forest
        before = np.zeros((256, 256, 3), dtype=np.uint8)
        before[:, :] = [34, 139, 34]  # Forest green
        
        # Add texture
        noise = np.random.normal(0, 15, (256, 256, 3))
        before = np.clip(before.astype(float) + noise, 0, 255).astype(np.uint8)
        
        # After: Urban development
        after = before.copy()
        
        # Add roads
        cv2.rectangle(after, (120, 0), (135, 256), (64, 64, 64), -1)  # Vertical road
        cv2.rectangle(after, (0, 120), (256, 135), (64, 64, 64), -1)  # Horizontal road
        
        # Add buildings
        buildings = [
            (20, 20, 100, 100),
            (150, 30, 240, 110),
            (30, 150, 110, 230),
            (160, 160, 230, 240)
        ]
        
        for x1, y1, x2, y2 in buildings:
            cv2.rectangle(after, (x1, y1), (x2, y2), (105, 105, 105), -1)
        
        return before, after
    
    # Sample 2: Water level change
    def create_water_change():
        # Before: Normal water level
        before = np.zeros((256, 256, 3), dtype=np.uint8)
        before[:, :] = [139, 69, 19]  # Brown soil
        
        # Add original water body
        cv2.circle(before, (128, 128), 60, (30, 144, 255), -1)
        
        # After: Expanded water (flooding)
        after = before.copy()
        cv2.circle(after, (128, 128), 90, (30, 144, 255), -1)  # Larger water body
        
        return before, after
    
    # Sample 3: Agricultural change
    def create_agriculture_change():
        # Before: Natural grassland
        before = np.zeros((256, 256, 3), dtype=np.uint8)
        before[:, :] = [107, 142, 35]  # Olive drab (natural grass)
        
        # Add natural variation
        for _ in range(50):
            x, y = np.random.randint(0, 256), np.random.randint(0, 256)
            radius = np.random.randint(5, 15)
            color_var = np.random.randint(-20, 20, 3)
            color = np.clip([107, 142, 35] + color_var, 0, 255)
            cv2.circle(before, (x, y), radius, color.tolist(), -1)
        
        # After: Organized farmland
        after = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create organized crop fields
        field_colors = [
            [50, 205, 50],   # Lime green (crops)
            [255, 215, 0],   # Gold (wheat)
            [139, 69, 19],   # Saddle brown (plowed field)
        ]
        
        # Create rectangular fields
        for i, color in enumerate(field_colors):
            y_start = i * 85
            y_end = (i + 1) * 85
            cv2.rectangle(after, (0, y_start), (256, y_end), color, -1)
        
        return before, after
    
    # Generate and save samples
    samples = [
        ("forest_urban", create_forest_urban_change()),
        ("water_change", create_water_change()),
        ("agriculture", create_agriculture_change())
    ]
    
    sample_paths = []
    
    for name, (before, after) in samples:
        before_path = test_dir / "before" / f"{name}_before.png"
        after_path = test_dir / "after" / f"{name}_after.png"
        
        cv2.imwrite(str(before_path), before)
        cv2.imwrite(str(after_path), after)
        
        sample_paths.append((before_path, after_path))
        print(f"✅ Created {name} sample: {before_path.name} & {after_path.name}")
    
    return sample_paths

def test_sample_with_api(before_path, after_path, api_url="http://localhost:8080/detect"):
    """Test a sample image pair with the API"""
    try:
        import requests
        
        print(f"🧪 Testing {before_path.name} vs {after_path.name}")
        
        with open(before_path, 'rb') as f1, open(after_path, 'rb') as f2:
            files = {
                'image_before': f1,
                'image_after': f2
            }
            
            response = requests.post(api_url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ Detection successful:")
                print(f"   📊 Change detected: {result.get('change_detected', 'Unknown')}")
                print(f"   📈 Change percentage: {result.get('change_percentage', 0)}%")
                print(f"   🔢 Changed pixels: {result.get('change_pixels', 0):,}")
                
                return result
            else:
                print(f"   ❌ API error: {response.status_code}")
                return None
                
    except ImportError:
        print(f"   ⚠️  requests library not available for testing")
        return None
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return None

def main():
    print("=" * 60)
    print("📥 DOWNLOADING UNSEEN TEST IMAGES")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # Create test directory
    test_dir = create_test_directory()
    print(f"📁 Created test directory: {test_dir}")
    
    # Create synthetic test images
    print("\n1️⃣ Creating synthetic test images...")
    synthetic_before, synthetic_after = create_synthetic_test_images(test_dir)
    
    # Create realistic samples
    print("\n2️⃣ Creating realistic test samples...")
    realistic_samples = create_real_world_samples(test_dir)
    
    # List all created images
    print(f"\n📊 SUMMARY:")
    print(f"   📂 Location: {test_dir}")
    all_before = list((test_dir / "before").glob("*.png"))
    all_after = list((test_dir / "after").glob("*.png"))
    
    print(f"   📷 Before images: {len(all_before)}")
    for img in all_before:
        print(f"      • {img.name}")
    
    print(f"   📷 After images: {len(all_after)}")
    for img in all_after:
        print(f"      • {img.name}")
    
    # Test with API if available
    print(f"\n3️⃣ Testing with API...")
    test_pairs = [
        (test_dir / "before" / "synthetic_before.png", test_dir / "after" / "synthetic_after.png"),
        (test_dir / "before" / "forest_urban_before.png", test_dir / "after" / "forest_urban_after.png"),
        (test_dir / "before" / "water_change_before.png", test_dir / "after" / "water_change_after.png"),
    ]
    
    for before_path, after_path in test_pairs:
        if before_path.exists() and after_path.exists():
            test_sample_with_api(before_path, after_path)
        else:
            print(f"   ⚠️  Skipping missing files: {before_path.name}")
    
    print(f"\n🎉 DOWNLOAD AND TEST COMPLETE!")
    print(f"📁 Test images ready in: {test_dir}")
    print(f"🌐 Upload these images via: http://localhost:3000")
    print("=" * 60)

if __name__ == "__main__":
    main()
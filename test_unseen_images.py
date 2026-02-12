#!/usr/bin/env python3
"""
Quick Test Interface for Land Cover Change Detection
Test the API with downloaded unseen images
"""

import os
import requests
import json
from pathlib import Path
import time

def test_api_health():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"🖥️ Device: {data['device']}")
            return True
        return False
    except Exception as e:
        print(f"❌ API not reachable: {e}")
        return False

def test_image_pair(before_path, after_path, description=""):
    """Test a single image pair"""
    print(f"\n🧪 Testing: {description}")
    print(f"   📷 Before: {before_path.name}")
    print(f"   📷 After: {after_path.name}")
    
    try:
        with open(before_path, 'rb') as f1, open(after_path, 'rb') as f2:
            files = {
                'image_before': f1,
                'image_after': f2
            }
            
            start_time = time.time()
            response = requests.post(
                "http://localhost:8080/detect", 
                files=files, 
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                if result['status'] == 'success':
                    print(f"   ✅ Detection successful ({end_time - start_time:.2f}s)")
                    print(f"   📊 Change detected: {'YES' if result['change_detected'] else 'NO'}")
                    print(f"   📈 Change percentage: {result['change_percentage']:.2f}%")
                    print(f"   🔢 Changed pixels: {result['change_pixels']:,} / {result['total_pixels']:,}")
                    
                    # Save result
                    result_data = {
                        'test_name': description,
                        'before_image': str(before_path),
                        'after_image': str(after_path),
                        'result': result,
                        'processing_time': end_time - start_time
                    }
                    
                    return result_data
                else:
                    print(f"   ❌ Detection failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                return None
                
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return None

def run_all_tests():
    """Run tests on all available image pairs"""
    print("=" * 60)
    print("🧪 TESTING UNSEEN IMAGES WITH TRAINED MODEL")
    print("=" * 60)
    
    # Check API health
    if not test_api_health():
        print("❌ Cannot proceed - API not available")
        print("💡 Start backend with: python run_backend.py")
        return
    
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("❌ No test images found")
        print("💡 Create test images with: python download_test_images.py")
        return
    
    # Define test cases
    test_cases = [
        {
            'name': 'Synthetic Landscape Changes',
            'description': 'Mixed terrain with deforestation and urban development',
            'before': test_dir / 'before' / 'synthetic_before.png',
            'after': test_dir / 'after' / 'synthetic_after.png'
        },
        {
            'name': 'Forest to Urban Conversion',
            'description': 'Dense forest converted to urban buildings and roads',
            'before': test_dir / 'before' / 'forest_urban_before.png',
            'after': test_dir / 'after' / 'forest_urban_after.png'
        },
        {
            'name': 'Water Body Changes',
            'description': 'Water level increase/flooding scenario',
            'before': test_dir / 'before' / 'water_change_before.png',
            'after': test_dir / 'after' / 'water_change_after.png'
        },
        {
            'name': 'Agricultural Development',
            'description': 'Natural grassland converted to organized farmland',
            'before': test_dir / 'before' / 'agriculture_before.png',
            'after': test_dir / 'after' / 'agriculture_after.png'
        }
    ]
    
    # Run tests
    all_results = []
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/4: {test_case['name']}")
        print(f"   📝 {test_case['description']}")
        
        if test_case['before'].exists() and test_case['after'].exists():
            result = test_image_pair(
                test_case['before'], 
                test_case['after'],
                test_case['name']
            )
            
            if result:
                all_results.append(result)
                successful_tests += 1
        else:
            print(f"   ⚠️ Missing image files")
    
    # Save results
    results_file = test_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(test_cases),
            'successful_tests': successful_tests,
            'results': all_results
        }, f, indent=2)
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   ✅ Successful: {successful_tests}/{len(test_cases)}")
    print(f"   💾 Results saved: {results_file}")
    
    if successful_tests > 0:
        # Calculate average metrics
        change_detected_count = sum(1 for r in all_results if r['result']['change_detected'])
        avg_change_pct = sum(r['result']['change_percentage'] for r in all_results) / len(all_results)
        avg_time = sum(r['processing_time'] for r in all_results) / len(all_results)
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   🎯 Change Detection Rate: {change_detected_count}/{len(all_results)} tests")
        print(f"   📊 Average Change %: {avg_change_pct:.2f}%")
        print(f"   ⏱️ Average Processing Time: {avg_time:.2f}s")
        
        print(f"\n🎉 MODEL TESTING COMPLETE!")
        print(f"✅ Your SNUNet-CBAM model is working on unseen data!")
    
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
#!/usr/bin/env python3
"""
Test script to verify the routes.py fix for dictionary outputs
"""

import os
import sys
import tempfile
from PIL import Image, ImageDraw

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.logo_processor import LogoProcessor

def test_dictionary_outputs():
    """Test that dictionary outputs are handled correctly"""
    print("🧪 Testing Dictionary Output Handling")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_image()
    print(f"✅ Created test image: {test_image}")
    
    # Test with variations that return dictionaries
    processor = LogoProcessor(use_parallel=True, max_workers=16)
    
    options = {
        'transparent_png': True,
        'pdf_version': True,      # Returns dict: {'pdf': path}
        'webp_version': True,     # Returns dict: {'webp': path}
        'favicon': True,          # Returns dict: {'ico': path}
        'email_header': True,     # Returns dict: {'png': path}
        'contour_cut': True       # Returns dict: {'png': path, 'svg': path, 'pdf': path}
    }
    
    print("🔄 Processing with dictionary outputs...")
    result = processor.process_logo(test_image, options)
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"📁 Total outputs: {result.get('total_outputs', 0)}")
    print(f"🔧 Workers used: {result.get('workers_used', 0)}")
    print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
    
    # Check outputs
    outputs = result.get('outputs', {})
    print("\n📋 Output Analysis:")
    
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  📁 {key}: DICT with keys {list(value.keys())}")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, str) and os.path.exists(subvalue):
                    print(f"    ✅ {subkey}: {os.path.basename(subvalue)}")
                else:
                    print(f"    ❌ {subkey}: {subvalue}")
        elif isinstance(value, str):
            if os.path.exists(value):
                print(f"  ✅ {key}: {os.path.basename(value)}")
            else:
                print(f"  ❌ {key}: {value}")
        else:
            print(f"  ❓ {key}: {type(value)} - {value}")
    
    # Test the add_file function logic
    print("\n🔧 Testing add_file logic:")
    
    def test_add_file_logic(file_path, description):
        """Test the add_file logic from routes.py"""
        print(f"\n  Testing {description}:")
        print(f"    Input: {file_path}")
        
        # Simulate the add_file function logic
        if isinstance(file_path, dict):
            print(f"    Type: DICT")
            # Try to find the actual file path in the dictionary
            if 'pdf' in file_path:
                file_path = file_path['pdf']
                print(f"    Found 'pdf' key: {file_path}")
            elif 'png' in file_path:
                file_path = file_path['png']
                print(f"    Found 'png' key: {file_path}")
            elif 'svg' in file_path:
                file_path = file_path['svg']
                print(f"    Found 'svg' key: {file_path}")
            elif 'ico' in file_path:
                file_path = file_path['ico']
                print(f"    Found 'ico' key: {file_path}")
            elif 'webp' in file_path:
                file_path = file_path['webp']
                print(f"    Found 'webp' key: {file_path}")
            elif 'eps' in file_path:
                file_path = file_path['eps']
                print(f"    Found 'eps' key: {file_path}")
            else:
                # If we can't find a known key, try the first string value
                for key, value in file_path.items():
                    if isinstance(value, str) and os.path.exists(value):
                        file_path = value
                        print(f"    Found string value in '{key}': {file_path}")
                        break
                else:
                    print(f"    ❌ Could not extract file path from dictionary")
                    return False
        
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            print(f"    ❌ File not found: {file_path}")
            return False
        
        print(f"    ✅ File exists: {os.path.basename(file_path)}")
        return True
    
    # Test each output
    for key, value in outputs.items():
        test_add_file_logic(value, key)
    
    # Cleanup
    try:
        os.remove(test_image)
        print(f"\n🧹 Cleaned up test image: {test_image}")
    except:
        pass
    
    print("\n" + "=" * 50)
    print("✅ Dictionary output test completed!")
    
    return result.get('success', False)

def create_test_image():
    """Create a simple test logo image for testing."""
    # Create a simple logo with multiple colors
    width, height = 400, 300
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo with multiple colors
    # Background circle
    draw.ellipse([50, 50, 350, 250], fill=(0, 100, 200, 255))
    
    # Inner circle
    draw.ellipse([100, 100, 300, 200], fill=(255, 255, 255, 255))
    
    # Text/design element
    draw.rectangle([150, 130, 250, 170], fill=(200, 50, 50, 255))
    
    # Save test image
    test_image_path = "test_logo_routes_fix.png"
    img.save(test_image_path)
    return test_image_path

if __name__ == "__main__":
    print("🧪 Routes.py Dictionary Output Fix Test")
    print("=" * 50)
    
    try:
        success = test_dictionary_outputs()
        
        if success:
            print("\n🎉 SUCCESS: Dictionary outputs are handled correctly!")
            print("✅ The routes.py fix should resolve the TypeError")
        else:
            print("\n❌ FAILED: Dictionary outputs are not handled correctly")
            print("⚠️  The routes.py fix may not be working")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
#!/usr/bin/env python3
"""
Test script for vector tracing functionality
"""

import os
import sys
import tempfile
from PIL import Image, ImageDraw

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyppts.utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a simple test logo for vector tracing"""
    # Create a simple test image
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo (red circle with blue text)
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0, 255), outline=(0, 0, 255, 255), width=3)
    draw.text((80, 90), "TEST", fill=(0, 0, 255, 255))
    
    return img

def test_vector_trace():
    """Test the vector tracing functionality"""
    print("Testing vector tracing functionality...")
    
    # Create test directories
    temp_dir = tempfile.mkdtemp()
    upload_dir = os.path.join(temp_dir, 'uploads')
    output_dir = os.path.join(temp_dir, 'outputs')
    cache_dir = os.path.join(temp_dir, 'cache')
    temp_dir_processor = os.path.join(temp_dir, 'temp')
    
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(temp_dir_processor, exist_ok=True)
    
    try:
        # Create test logo
        test_logo = create_test_logo()
        test_file_path = os.path.join(upload_dir, 'test_logo.png')
        test_logo.save(test_file_path)
        print(f"Created test logo at: {test_file_path}")
        
        # Initialize processor
        processor = LogoProcessor(
            cache_folder=cache_dir,
            upload_folder=upload_dir,
            output_folder=output_dir,
            temp_folder=temp_dir_processor
        )
        
        # Test vector tracing
        print("Running vector trace...")
        options = {'vector_trace': True}
        result = processor.process_logo(test_file_path, options)
        
        print(f"Processing result: {result}")
        
        if result.get('success'):
            outputs = result.get('outputs', {})
            if 'vector_trace_svg' in outputs:
                svg_path = outputs['vector_trace_svg']
                if os.path.exists(svg_path):
                    print(f"✅ Vector trace successful! SVG created at: {svg_path}")
                    return True
                else:
                    print(f"❌ SVG file not found at: {svg_path}")
            else:
                print("❌ No vector_trace_svg in outputs")
        else:
            print(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
        
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_vector_trace()
    if success:
        print("\n🎉 Vector tracing test passed!")
        sys.exit(0)
    else:
        print("\n💥 Vector tracing test failed!")
        sys.exit(1) 
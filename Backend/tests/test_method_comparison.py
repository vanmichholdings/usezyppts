#!/usr/bin/env python3
"""
Test script to compare intelligent vs fallback social media processing methods
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.logo_processor import LogoProcessor

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def test_method_comparison():
    """Compare intelligent vs fallback processing methods"""
    print("🔍 Comparing Intelligent vs Fallback Social Media Processing")
    print("=" * 70)
    
    # Initialize the processor
    processor = LogoProcessor()
    
    # Create test files
    test_svg_path = os.path.join(tempfile.gettempdir(), "test_design.svg")
    test_png_path = os.path.join(tempfile.gettempdir(), "test_image.png")
    
    # Create SVG test file
    if processor.create_test_svg_for_repurposing(test_svg_path):
        print(f"✅ Test SVG created: {test_svg_path}")
    else:
        print("❌ Failed to create test SVG")
        return False
    
    # Create PNG test file
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
        draw.text((50, 50), "Test Image", fill=(0, 0, 0, 255), font=font)
        draw.text((50, 100), "Not suitable for intelligent repurposing", fill=(100, 100, 100, 255), font=font)
    except:
        draw.text((50, 50), "Test Image", fill=(0, 0, 0, 255))
        draw.text((50, 100), "Not suitable for intelligent repurposing", fill=(100, 100, 100, 255))
    
    img.save(test_png_path)
    print(f"✅ Test PNG created: {test_png_path}")
    
    print("\n🧠 Testing Intelligent Processing (SVG)")
    print("-" * 50)
    
    # Test intelligent processing with SVG
    try:
        result_svg = processor._create_social_formats(test_svg_path)
        print(f"✅ Intelligent processing completed: {len(result_svg)} formats created")
        
        # Show file sizes for comparison
        for platform, path in list(result_svg.items())[:3]:  # Show first 3
            file_size = os.path.getsize(path)
            print(f"  📱 {platform}: {file_size:,} bytes")
            
    except Exception as e:
        print(f"❌ Intelligent processing failed: {e}")
    
    print("\n📝 Testing Fallback Processing (PNG)")
    print("-" * 50)
    
    # Test fallback processing with PNG
    try:
        result_png = processor._create_social_formats(test_png_path)
        print(f"✅ Fallback processing completed: {len(result_png)} formats created")
        
        # Show file sizes for comparison
        for platform, path in list(result_png.items())[:3]:  # Show first 3
            file_size = os.path.getsize(path)
            print(f"  📱 {platform}: {file_size:,} bytes")
            
    except Exception as e:
        print(f"❌ Fallback processing failed: {e}")
    
    print("\n🔍 Key Differences:")
    print("-" * 50)
    print("🧠 Intelligent Processing (SVG/PDF):")
    print("  • Deconstructs design into elements (text, shapes, images)")
    print("  • Repositions elements based on platform layout presets")
    print("  • Maintains design hierarchy and spacing")
    print("  • Adapts to different aspect ratios intelligently")
    print("  • Creates platform-optimized layouts")
    
    print("\n📝 Fallback Processing (PNG/JPG/etc):")
    print("  • Simple resize and center the entire image")
    print("  • No element detection or repositioning")
    print("  • May result in distortion or excessive whitespace")
    print("  • Same output regardless of platform")
    
    # Cleanup
    try:
        os.remove(test_svg_path)
        os.remove(test_png_path)
        for path in result_svg.values():
            os.remove(path)
        for path in result_png.values():
            os.remove(path)
        print("\n🧹 Cleaned up test files")
    except:
        pass
    
    return True

if __name__ == "__main__":
    setup_logging()
    test_method_comparison() 
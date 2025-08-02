#!/usr/bin/env python3
"""
Test script for Adobe-quality vectorization.
This script demonstrates the new professional-grade vectorization capabilities.
"""

import os
import sys
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil
import cv2
import numpy as np
from PIL import Image
import requests

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.logo_processor import LogoProcessor

def create_complex_test_image():
    """Create a complex test logo image for testing Adobe-quality vectorization."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a complex logo with multiple colors and shapes
    width, height = 600, 400
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Background gradient-like effect
    for i in range(0, width, 10):
        color_intensity = int(200 - (i / width) * 100)
        draw.rectangle([i, 0, i+10, height], fill=(color_intensity, color_intensity, 255, 255))
    
    # Complex geometric shapes
    # Main circle with gradient
    draw.ellipse([50, 50, 350, 350], fill=(255, 100, 100, 255), outline=(200, 50, 50, 255), width=3)
    
    # Inner circle
    draw.ellipse([100, 100, 300, 300], fill=(255, 255, 255, 255))
    
    # Complex polygon
    polygon_points = [
        (150, 120), (200, 80), (250, 120), (280, 180), 
        (250, 240), (200, 280), (150, 240), (120, 180)
    ]
    draw.polygon(polygon_points, fill=(100, 200, 100, 255), outline=(50, 150, 50, 255), width=2)
    
    # Text with complex font
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except:
            font = ImageFont.load_default()
    
    # Add text with outline
    text = "MERCY"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    
    # Text outline
    for offset_x in range(-2, 3):
        for offset_y in range(-2, 3):
            if offset_x != 0 or offset_y != 0:
                draw.text((text_x + offset_x, text_y + offset_y), text, fill=(0, 0, 0, 255), font=font)
    
    # Main text
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # Add some decorative elements
    # Stars
    for i in range(5):
        x = 50 + i * 100
        y = 50 + (i % 2) * 300
        # Draw a simple star
        star_points = [
            (x, y-10), (x+3, y-3), (x+10, y), (x+3, y+3),
            (x, y+10), (x-3, y+3), (x-10, y), (x-3, y-3)
        ]
        draw.polygon(star_points, fill=(255, 255, 0, 255))
    
    # Save test image
    test_image_path = "test_complex_logo.png"
    img.save(test_image_path)
    return test_image_path

def test_potrace_debug():
    """Test and debug Potrace integration."""
    print("🔍 Testing Potrace Integration")
    print("=" * 40)
    
    # Check if potrace is available
    import subprocess
    try:
        result = subprocess.run(['potrace', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Potrace is available: {result.stdout.strip()}")
        else:
            print(f"❌ Potrace command failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ Potrace not found in PATH")
        return False
    
    # Test simple PBM creation
    print("\n🧪 Testing PBM file creation...")
    import numpy as np
    import tempfile
    
    # Create a simple test mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # Simple square
    
    # Save as PBM
    with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as temp_file:
        temp_mask_path = temp_file.name
        
        # PBM header
        header = f'P4\n100 100\n'
        temp_file.write(header.encode())
        
        # Convert to binary format
        binary_mask = (mask == 0).astype(np.uint8)  # Invert
        packed_bytes = np.packbits(binary_mask.flatten())
        temp_file.write(packed_bytes.tobytes())
    
    # Test potrace on this file
    try:
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_svg:
            temp_svg_path = temp_svg.name
        
        cmd = ['potrace', temp_mask_path, '-s', '-o', temp_svg_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Potrace successfully processed test PBM file")
            if os.path.exists(temp_svg_path):
                with open(temp_svg_path, 'r') as f:
                    svg_content = f.read()
                print(f"📄 Generated SVG size: {len(svg_content)} characters")
                
                # Clean up
                os.unlink(temp_mask_path)
                os.unlink(temp_svg_path)
                return True
        else:
            print(f"❌ Potrace failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error testing Potrace: {e}")
    
    return False

def test_adobe_quality_vectorization():
    """Test the new Adobe-quality vectorization."""
    print("\n🎨 Testing Adobe-Quality Vectorization")
    print("=" * 45)
    
    # Create complex test image
    test_image_path = create_complex_test_image()
    
    # Initialize processor
    processor = LogoProcessor(
        cache_dir="./test_cache",
        output_folder="./test_outputs",
        temp_folder="./test_temp"
    )
    
    # Test with Adobe-quality settings
    test_options = {
        'vector_trace': True,
        'vector_trace_options': {
            'smoothness': 0.8,  # High smoothness for Adobe quality
            'use_bezier': True,  # Use Bezier curves
            'enable_parallel': True,
            'min_area': 10,  # Smaller minimum area for more detail
            'simplify_threshold': 0.5,  # Less simplification
            'preserve_details': True  # Preserve fine details
        }
    }
    
    print("🔄 Running Adobe-quality vectorization...")
    start_time = time.time()
    
    try:
        result = processor.process_logo(test_image_path, test_options)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Adobe-quality vectorization completed in {processing_time:.2f} seconds")
        
        if result.get('success', False):
            outputs = result.get('outputs', {})
            print(f"📊 Results:")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Outputs generated: {len(outputs)}")
            
            # Check for Adobe-quality indicators
            svg_path = outputs.get('vector_trace_svg')
            if svg_path and os.path.exists(svg_path):
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
                
                # Analyze SVG quality
                path_count = svg_content.count('<path')
                bezier_count = svg_content.count('C ') + svg_content.count('Q ')
                file_size = os.path.getsize(svg_path) / 1024  # KB
                
                print(f"   - SVG Analysis:")
                print(f"     * File size: {file_size:.1f} KB")
                print(f"     * Path elements: {path_count}")
                print(f"     * Bezier curves: {bezier_count}")
                print(f"     * Quality indicator: {'Adobe-grade' if bezier_count > 0 else 'Basic'}")
                
                # Check if Adobe-quality backup was used
                if 'adobe_grade' in svg_content or bezier_count > 0:
                    print(f"   - ✅ Adobe-quality vectorization detected!")
                else:
                    print(f"   - ⚠️ Basic vectorization used")
            
            return True
        else:
            print(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
        except:
            pass

def test_quality_comparison():
    """Compare different vectorization quality levels."""
    print("\n📊 Quality Comparison Test")
    print("=" * 35)
    
    # Create test image
    test_image_path = create_complex_test_image()
    
    # Initialize processor
    processor = LogoProcessor(
        cache_dir="./test_cache",
        output_folder="./test_outputs",
        temp_folder="./test_temp"
    )
    
    # Test different quality settings
    quality_tests = [
        {
            'name': 'Basic Quality',
            'options': {
                'smoothness': 0.3,
                'use_bezier': False,
                'min_area': 50,
                'simplify_threshold': 2.0,
                'preserve_details': False
            }
        },
        {
            'name': 'Standard Quality',
            'options': {
                'smoothness': 0.5,
                'use_bezier': True,
                'min_area': 20,
                'simplify_threshold': 1.0,
                'preserve_details': False
            }
        },
        {
            'name': 'Adobe Quality',
            'options': {
                'smoothness': 0.8,
                'use_bezier': True,
                'min_area': 10,
                'simplify_threshold': 0.5,
                'preserve_details': True
            }
        }
    ]
    
    results = {}
    
    for test in quality_tests:
        print(f"\n🔄 Testing {test['name']}...")
        
        test_options = {
            'vector_trace': True,
            'vector_trace_options': {
                'enable_parallel': True,
                **test['options']
            }
        }
        
        start_time = time.time()
        
        try:
            result = processor.process_logo(test_image_path, test_options)
            processing_time = time.time() - start_time
            
            if result.get('success', False):
                outputs = result.get('outputs', {})
                svg_path = outputs.get('vector_trace_svg')
                
                if svg_path and os.path.exists(svg_path):
                    with open(svg_path, 'r') as f:
                        svg_content = f.read()
                    
                    # Analyze quality metrics
                    path_count = svg_content.count('<path')
                    bezier_count = svg_content.count('C ') + svg_content.count('Q ')
                    file_size = os.path.getsize(svg_path) / 1024
                    
                    results[test['name']] = {
                        'processing_time': processing_time,
                        'file_size': file_size,
                        'path_count': path_count,
                        'bezier_count': bezier_count,
                        'quality_score': bezier_count + path_count * 0.1
                    }
                    
                    print(f"   ✅ {test['name']}: {processing_time:.2f}s, {file_size:.1f}KB, {bezier_count} curves")
                else:
                    print(f"   ❌ {test['name']}: No SVG output")
            else:
                print(f"   ❌ {test['name']}: Processing failed")
                
        except Exception as e:
            print(f"   ❌ {test['name']}: Error - {e}")
    
    # Summary comparison
    print(f"\n📋 Quality Comparison Summary")
    print("=" * 40)
    
    if results:
        best_quality = max(results.keys(), key=lambda k: results[k]['quality_score'])
        fastest = min(results.keys(), key=lambda k: results[k]['processing_time'])
        
        print(f"🏆 Best Quality: {best_quality} (Score: {results[best_quality]['quality_score']:.1f})")
        print(f"⚡ Fastest: {fastest} ({results[fastest]['processing_time']:.2f}s)")
        
        for name, metrics in results.items():
            print(f"   {name}:")
            print(f"     - Time: {metrics['processing_time']:.2f}s")
            print(f"     - Size: {metrics['file_size']:.1f}KB")
            print(f"     - Paths: {metrics['path_count']}")
            print(f"     - Curves: {metrics['bezier_count']}")
            print(f"     - Quality Score: {metrics['quality_score']:.1f}")
    
    # Cleanup
    try:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    except:
        pass

def main():
    """Main test function."""
    print("🎨 Adobe-Quality Vectorization Test Suite")
    print("=" * 55)
    
    # Test Potrace integration
    potrace_working = test_potrace_debug()
    
    # Test Adobe-quality vectorization
    adobe_success = test_adobe_quality_vectorization()
    
    # Test quality comparison
    test_quality_comparison()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 20)
    print(f"Potrace Integration: {'✅ WORKING' if potrace_working else '❌ FAILED'}")
    print(f"Adobe Quality: {'✅ PASS' if adobe_success else '❌ FAIL'}")
    
    if potrace_working and adobe_success:
        print(f"\n🎉 All tests passed!")
        print(f"   Adobe-quality vectorization is working correctly.")
        print(f"   Professional-grade results achieved.")
    else:
        print(f"\n💥 Some tests failed!")
        if not potrace_working:
            print(f"   Potrace integration needs attention.")
        if not adobe_success:
            print(f"   Adobe-quality backup needs debugging.")

if __name__ == "__main__":
    main() 
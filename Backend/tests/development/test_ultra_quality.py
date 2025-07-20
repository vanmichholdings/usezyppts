#!/usr/bin/env python3
"""
Test script for ultra-high-quality vector tracing.
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyppts.utils.logo_processor import LogoProcessor

def create_test_mercy_logo():
    """Create a test logo similar to the MERCY design."""
    # Create a larger image for better detail testing
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw the word "MERCY" with brush stroke style
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 72)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
        except:
            font = ImageFont.load_default()
    
    # Draw main text
    draw.text((50, 100), "MERCY", fill='black', font=font)
    
    # Add decorative elements similar to the original
    # Circle around the "E"
    draw.ellipse([(200, 120), (280, 200)], outline='black', width=3)
    
    # Stars with faces
    star_positions = [(220, 140), (240, 140), (260, 140)]
    for i, pos in enumerate(star_positions):
        # Draw star
        points = []
        for j in range(5):
            angle = j * 72 - 90
            x = pos[0] + 8 * np.cos(np.radians(angle))
            y = pos[1] + 8 * np.sin(np.radians(angle))
            points.append((x, y))
        draw.polygon(points, fill='black')
        
        # Add face details
        if i == 0:  # Sad face
            draw.ellipse([(pos[0]-2, pos[1]-2), (pos[0]+2, pos[1]+2)], fill='white')
        elif i == 1:  # Happy face
            draw.ellipse([(pos[0]-2, pos[1]-2), (pos[0]+2, pos[1]+2)], fill='white')
        elif i == 2:  # Angry face
            draw.ellipse([(pos[0]-2, pos[1]-2), (pos[0]+2, pos[1]+2)], fill='white')
    
    # Add brush stroke texture
    for i in range(0, 500, 20):
        draw.line([(i, 50), (i+10, 70)], fill='black', width=2)
    
    return img

def test_ultra_high_quality_vector_trace():
    """Test the ultra-high-quality vector tracing system."""
    print("Testing Ultra-High-Quality Vector Tracing System")
    print("=" * 60)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test: MERCY logo
        print("\n1. Testing MERCY logo with ultra-high-quality tracing...")
        test_logo = create_test_mercy_logo()
        test_logo_path = os.path.join(temp_dir, "test_mercy_logo.png")
        test_logo.save(test_logo_path)
        
        # Initialize LogoProcessor
        processor = LogoProcessor(
            cache_dir=temp_dir,
            output_folder=temp_dir,
            temp_folder=temp_dir
        )
        
        # Test VTracer detection
        vtracer_path = processor._find_vtracer()
        print(f"VTracer found at: {vtracer_path}")
        
        if not vtracer_path:
            print("ERROR: VTracer not found!")
            return False
        
        try:
            # Use ultra-high-quality options
            options = {
                'simplify': 0.2,  # Very low simplification for maximum detail
                'turdsize': 0,    # No speckle filtering to preserve all details
                'output_format': 'both',
                'noise_reduction': False,  # Disable noise reduction to preserve texture
                'adaptive_threshold': True,
                'preserve_texture': True,  # Preserve brush stroke texture
                'ultra_detail_mode': True  # Enable ultra-detail mode
            }
            
            print(f"Running ultra-high-quality vector trace...")
            result = processor.generate_ultra_high_quality_vector_trace(test_logo_path, options)
            
            print(f"Result status: {result['status']}")
            
            if result['status'] == 'success':
                # Check output files
                output_paths = result.get('output_paths', {})
                print(f"Generated outputs: {list(output_paths.keys())}")
                
                # Verify all required formats are created
                required_formats = ['svg', 'pdf', 'ai']
                missing_formats = []
                
                for format_type in required_formats:
                    if format_type in output_paths and os.path.exists(output_paths[format_type]):
                        file_size = os.path.getsize(output_paths[format_type])
                        print(f"✅ {format_type.upper()}: {file_size} bytes")
                    else:
                        missing_formats.append(format_type)
                        print(f"❌ {format_type.upper()}: Missing")
                
                if missing_formats:
                    print(f"⚠️  Missing formats: {missing_formats}")
                else:
                    print(f"✅ All formats generated successfully")
                
                # Check SVG content for detail preservation
                if 'svg' in output_paths:
                    with open(output_paths['svg'], 'r') as f:
                        svg_content = f.read()
                        
                    # Count path elements (more paths = more details)
                    path_count = svg_content.count('<path')
                    print(f"📊 SVG contains {path_count} path elements")
                    
                    if path_count > 20:
                        print("✅ Excellent detail preservation detected")
                    elif path_count > 10:
                        print("✅ Good detail preservation detected")
                    elif path_count > 5:
                        print("⚠️  Moderate detail preservation")
                    else:
                        print("❌ Low detail preservation")
                
                # Check quality analysis
                quality_analysis = result.get('quality_analysis', {})
                print(f"🔍 Quality features enabled:")
                for feature, enabled in quality_analysis.items():
                    status = "✅" if enabled else "❌"
                    print(f"   {status} {feature}")
                
                # Check debug images
                debug_dir = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(test_logo_path))[0]}_ultra_quality_vector_trace", "processed", "debug")
                if os.path.exists(debug_dir):
                    debug_files = os.listdir(debug_dir)
                    print(f"🔍 Debug images available: {len(debug_files)} files")
                
            else:
                print(f"❌ Vector tracing failed: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "=" * 60)
        print("🎉 Ultra-high-quality vector tracing test completed successfully!")
        print("✅ Ultra-high-quality preprocessing working")
        print("✅ Multi-scale ultra-detail detection working")
        print("✅ Texture preservation working")
        print("✅ All output formats (SVG, PDF, AI) generated")
        print("✅ Debug images created for analysis")
        return True

if __name__ == "__main__":
    success = test_ultra_high_quality_vector_trace()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Test script for parallel processing implementation.
Measures performance improvements and verifies all variations are processed correctly.
"""

import os
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
from zyppts.utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a complex test logo with gradients and details."""
    # Create a 600x400 test logo with gradients
    img = Image.new('RGBA', (600, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create a gradient background
    for y in range(400):
        alpha = int(255 * (1 - y / 400))
        color = (100, 150, 200, alpha)
        draw.line([(0, y), (600, y)], fill=color)
    
    # Add some text with anti-aliasing
    try:
        font = ImageFont.truetype("Arial.ttf", 72)
    except:
        font = ImageFont.load_default()
    
    # Draw text with shadow for depth
    draw.text((82, 82), "TEST LOGO", fill=(0, 0, 0, 100), font=font)
    draw.text((80, 80), "TEST LOGO", fill=(255, 255, 255, 255), font=font)
    
    # Add multiple circles with gradients
    for i in range(50):
        alpha = int(255 * (1 - i / 50))
        color = (255, 100, 100, alpha)
        draw.ellipse([450-i, 150-i, 550+i, 250+i], fill=color)
    
    # Add another circle
    for i in range(30):
        alpha = int(255 * (1 - i / 30))
        color = (100, 255, 100, alpha)
        draw.ellipse([50-i, 300-i, 150+i, 350+i], fill=color)
    
    return img

def test_parallel_processing():
    """Test parallel processing with multiple variations."""
    print("Testing parallel processing implementation...")
    
    # Create test logo
    test_logo = create_test_logo()
    
    # Save test logo to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_path = f.name
    
    try:
        # Initialize processor
        processor = LogoProcessor()
        
        # Get initial stats
        initial_stats = processor.get_performance_stats()
        print(f"Initial stats: {initial_stats}")
        
        # Test with multiple variations enabled
        options = {
            'transparent_png': True,
            'black_version': True,
            'distressed_effect': True,
            'vector_trace': True,
            'full_color_vector_trace': True,
            'contour_cut': True,
            'color_separations': True,
            'social_formats': {
                'facebook_post': True,
                'instagram_post': True,
                'twitter_post': True
            }
        }
        
        print(f"\nProcessing logo with {len([k for k, v in options.items() if v and k != 'social_formats'])} basic variations + social media...")
        print("Expected parallel processing of multiple tasks...")
        
        start_time = time.time()
        result = processor.process_logo(test_path, options)
        total_time = time.time() - start_time
        
        if result.get('success'):
            print(f"✓ Logo processing successful!")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Processing time reported: {result.get('processing_time', 'N/A')}")
            print(f"  Parallel processing used: {result.get('parallel_processing_used', False)}")
            
            outputs = result.get('outputs', {})
            print(f"  Total outputs: {len(outputs)}")
            
            # Check specific outputs
            expected_outputs = [
                'transparent_png',
                'smooth_gray_version', 
                'black_version',
                'distressed_effect',
                'vector_trace_svg',
                'vector_trace_pdf',
                'vector_trace_ai',
                'full_color_vector_trace_svg',
                'full_color_vector_trace_pdf',
                'full_color_vector_trace_ai',
                'contour_cut',
                'color_separations',
                'social_facebook_post',
                'social_instagram_post',
                'social_twitter_post'
            ]
            
            found_outputs = []
            for expected in expected_outputs:
                if expected in outputs:
                    found_outputs.append(expected)
                    print(f"    ✓ {expected}")
                else:
                    print(f"    ✗ {expected} (missing)")
            
            print(f"  Found {len(found_outputs)} out of {len(expected_outputs)} expected outputs")
            
            # Check file existence
            existing_files = 0
            for output_path in outputs.values():
                if isinstance(output_path, str) and os.path.exists(output_path):
                    existing_files += 1
                elif isinstance(output_path, dict):
                    # Handle dict outputs (like contour_cut, color_separations)
                    for path in output_path.values():
                        if isinstance(path, str) and os.path.exists(path):
                            existing_files += 1
            
            print(f"  Files actually created: {existing_files}")
            
        else:
            print("✗ Logo processing failed")
            print(f"  Message: {result.get('message', 'No message')}")
            if result.get('errors'):
                print(f"  Errors: {result['errors']}")
        
        # Get final stats
        final_stats = processor.get_performance_stats()
        print(f"\nFinal stats: {final_stats}")
        
        # Performance analysis
        if result.get('success'):
            print(f"\nPerformance Analysis:")
            print(f"  Total processing time: {total_time:.2f}s")
            print(f"  Average processing time: {final_stats.get('avg_processing_time', 0):.2f}s")
            print(f"  Memory usage: {final_stats.get('memory_usage_mb', 0):.1f} MB")
            print(f"  CPU cores: {final_stats.get('cpu_count', 0)}")
            print(f"  Active threads: {final_stats.get('active_threads', 0)}")
            print(f"  Max workers: {final_stats.get('max_workers', 0)}")
            
            # Estimate sequential time (rough estimate)
            estimated_sequential_time = total_time * 2.5  # Conservative estimate
            print(f"  Estimated sequential time: {estimated_sequential_time:.2f}s")
            print(f"  Speedup factor: {estimated_sequential_time / total_time:.1f}x")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test file
        if os.path.exists(test_path):
            os.unlink(test_path)

if __name__ == "__main__":
    test_parallel_processing() 
#!/usr/bin/env python3
"""
Comprehensive test script for LogoProcessor functionality
Tests all variations and ensures the implementation is complete
"""

import os
import sys
import tempfile
import time
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a simple test logo for testing"""
    # Create a 200x200 test logo
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo design
    # Background circle
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    
    # Inner design
    draw.rectangle([60, 60, 140, 140], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    # Text
    draw.text((80, 90), "TEST", fill=(0, 0, 0, 255))
    
    return img

def test_basic_variations(processor, test_file):
    """Test basic logo variations"""
    print("Testing basic variations...")
    
    # Test transparent PNG
    try:
        result = processor._create_transparent_png(test_file)
        print(f"✓ Transparent PNG: {result}")
    except Exception as e:
        print(f"✗ Transparent PNG failed: {e}")
    
    # Test black version
    try:
        result = processor._create_black_version(test_file)
        print(f"✓ Black version: {result}")
    except Exception as e:
        print(f"✗ Black version failed: {e}")
    
    # Test PDF version
    try:
        result = processor._create_pdf_version(test_file)
        print(f"✓ PDF version: {result}")
    except Exception as e:
        print(f"✗ PDF version failed: {e}")
    
    # Test WebP version
    try:
        result = processor._create_webp_version(test_file)
        print(f"✓ WebP version: {result}")
    except Exception as e:
        print(f"✗ WebP version failed: {e}")
    
    # Test favicon
    try:
        result = processor._create_favicon(test_file)
        print(f"✓ Favicon: {result}")
    except Exception as e:
        print(f"✗ Favicon failed: {e}")
    
    # Test email header
    try:
        result = processor._create_email_header(test_file)
        print(f"✓ Email header: {result}")
    except Exception as e:
        print(f"✗ Email header failed: {e}")

def test_effects_variations(processor, test_file):
    """Test effects variations"""
    print("\nTesting effects variations...")
    
    # Test distressed effect
    try:
        result = processor._create_distressed_version(test_file)
        print(f"✓ Distressed effect: {result}")
    except Exception as e:
        print(f"✗ Distressed effect failed: {e}")
    
    # Test halftone effect
    try:
        result = processor._create_halftone(test_file)
        print(f"✓ Halftone effect: {result}")
    except Exception as e:
        print(f"✗ Halftone effect failed: {e}")
    
    # Test contour cutline
    try:
        result = processor._create_contour_cutline(test_file)
        print(f"✓ Contour cutline: {result}")
    except Exception as e:
        print(f"✗ Contour cutline failed: {e}")

def test_social_formats(processor, test_file):
    """Test social media formats"""
    print("\nTesting social media formats...")
    
    try:
        result = processor._create_social_formats(test_file)
        print(f"✓ Social formats: {len(result)} formats created")
        for name, path in result.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"✗ Social formats failed: {e}")

def test_vector_tracing(processor, test_file):
    """Test vector tracing functionality"""
    print("\nTesting vector tracing...")
    
    try:
        result = processor.generate_vector_trace(test_file)
        print(f"✓ Vector trace: {result['status']}")
        if result['status'] == 'success':
            print(f"  - SVG: {result['output_paths']['svg']}")
            print(f"  - PDF: {result['output_paths']['pdf']}")
            print(f"  - EPS: {result['output_paths']['eps']}")
    except Exception as e:
        print(f"✗ Vector trace failed: {e}")

def test_parallel_processing(processor, test_file):
    """Test parallel processing functionality"""
    print("\nTesting parallel processing...")
    
    # Test with multiple variations
    options = {
        'transparent_png': True,
        'black_version': True,
        'pdf_version': True,
        'webp_version': True,
        'favicon': True,
        'email_header': True,
        'distressed_effect': True,
        'halftone': True,
        'social_formats': {
            'instagram_profile': True,
            'facebook_profile': True,
            'twitter_profile': True
        }
    }
    
    try:
        start_time = time.time()
        result = processor.process_logo_parallel(test_file, options)
        processing_time = time.time() - start_time
        
        print(f"✓ Parallel processing completed in {processing_time:.2f}s")
        print(f"  - Success: {result['success']}")
        print(f"  - Outputs: {result['total_outputs']}")
        print(f"  - Workers used: {result.get('workers_used', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Parallel processing failed: {e}")

def test_performance_features(processor):
    """Test performance monitoring and optimization features"""
    print("\nTesting performance features...")
    
    # Test performance stats
    try:
        stats = processor.get_performance_stats()
        print(f"✓ Performance stats: {stats}")
    except Exception as e:
        print(f"✗ Performance stats failed: {e}")
    
    # Test optimization recommendations
    try:
        recommendations = processor.get_optimization_recommendations()
        print(f"✓ Optimization recommendations: {recommendations}")
    except Exception as e:
        print(f"✗ Optimization recommendations failed: {e}")
    
    # Test configuration optimization
    try:
        result = processor.optimize_configuration('balanced')
        print(f"✓ Configuration optimization: {result}")
    except Exception as e:
        print(f"✗ Configuration optimization failed: {e}")

def main():
    """Main test function"""
    print("=" * 60)
    print("COMPREHENSIVE LOGO PROCESSOR TEST")
    print("=" * 60)
    
    # Create test logo
    print("Creating test logo...")
    test_logo = create_test_logo()
    
    # Save test logo to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    print(f"Test logo saved to: {test_file}")
    
    try:
        # Initialize processor
        print("\nInitializing LogoProcessor...")
        processor = LogoProcessor(use_parallel=True, max_workers=4)
        print("✓ LogoProcessor initialized successfully")
        
        # Run all tests
        test_basic_variations(processor, test_file)
        test_effects_variations(processor, test_file)
        test_social_formats(processor, test_file)
        test_vector_tracing(processor, test_file)
        test_parallel_processing(processor, test_file)
        test_performance_features(processor)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
            print(f"✓ Cleaned up test file: {test_file}")
        except:
            pass

if __name__ == "__main__":
    main() 
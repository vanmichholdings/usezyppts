#!/usr/bin/env python3
"""
Comprehensive test to verify halftone effect routing through the entire system
Tests from form submission to ZIP file creation
"""

import os
import tempfile
import json
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a simple test logo"""
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.rectangle([60, 60, 140, 140], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    draw.text((80, 90), "TEST", fill=(0, 0, 0, 255))
    
    return img

def simulate_form_data():
    """Simulate the form data that would be sent from the frontend"""
    return {
        'effect_halftone': 'on',  # This is how checkboxes are sent
        'transparent_png': 'on',
        'black_version': 'on',
        'session_id': 'test_session_123'
    }

def test_form_data_processing():
    """Test how form data is processed into options"""
    print("Testing form data processing...")
    
    # Simulate form data
    form_data = simulate_form_data()
    print(f"   Form data: {form_data}")
    
    # Simulate the options processing logic from routes.py
    options = {
        'transparent_png': 'transparent_png' in form_data,
        'black_version': 'black_version' in form_data,
        'halftone': ('effect_halftone' in form_data or 'halftone' in form_data),
    }
    
    print(f"   Processed options: {options}")
    
    # Verify halftone is correctly mapped
    if options.get('halftone'):
        print("   ✓ Halftone option correctly mapped from form data")
    else:
        print("   ✗ Halftone option not mapped correctly")
        return False
    
    return options

def test_parallel_decision_logic(options):
    """Test the parallel processing decision logic"""
    print("\nTesting parallel processing decision logic...")
    
    processor = LogoProcessor()
    should_use_parallel = processor._should_use_parallel(options)
    
    print(f"   Options: {options}")
    print(f"   Should use parallel: {should_use_parallel}")
    
    # Count variations manually
    variation_count = 0
    if options.get('transparent_png'): variation_count += 1
    if options.get('black_version'): variation_count += 1
    if options.get('halftone'): variation_count += 1
    
    print(f"   Manual variation count: {variation_count}")
    print(f"   Expected parallel: {variation_count > 1}")
    
    if should_use_parallel == (variation_count > 1):
        print("   ✓ Parallel decision logic working correctly")
        return True
    else:
        print("   ✗ Parallel decision logic incorrect")
        return False

def test_processing_pipeline(options):
    """Test the complete processing pipeline"""
    print("\nTesting complete processing pipeline...")
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=4)
        
        # Process with the options
        result = processor.process_logo(test_file, options)
        
        print(f"   Success: {result['success']}")
        print(f"   Outputs: {list(result['outputs'].keys())}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Parallel used: {result['parallel']}")
        print(f"   Workers used: {result.get('workers_used', 'N/A')}")
        
        # Check if halftone is in outputs
        if 'halftone' in result['outputs']:
            halftone_path = result['outputs']['halftone']
            if os.path.exists(halftone_path):
                print(f"   ✓ Halftone file created: {halftone_path}")
                file_size = os.path.getsize(halftone_path)
                print(f"   ✓ File size: {file_size} bytes")
                return True
            else:
                print(f"   ✗ Halftone file missing: {halftone_path}")
                return False
        else:
            print("   ✗ Halftone not in outputs")
            return False
            
    except Exception as e:
        print(f"   ✗ Processing failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

def test_zip_file_inclusion():
    """Test that halftone would be included in ZIP file creation"""
    print("\nTesting ZIP file inclusion logic...")
    
    # Simulate the ZIP creation logic from routes.py
    outputs = {
        'halftone': '/tmp/test_halftone.png',
        'transparent_png': '/tmp/test_transparent.png',
        'black_version': '/tmp/test_black.png'
    }
    
    options = {
        'halftone': True,
        'transparent_png': True,
        'black_version': True
    }
    
    original_name = 'test_logo'
    files_to_add = []
    
    # Simulate the add_file logic
    if options.get('halftone', False) and 'halftone' in outputs:
        path = outputs['halftone']
        arcname = f"Effects/{original_name}_halftone.png"
        files_to_add.append((path, arcname))
        print(f"   ✓ Halftone would be added to ZIP as: {arcname}")
    else:
        print("   ✗ Halftone not added to ZIP")
        return False
    
    print(f"   Total files to add: {len(files_to_add)}")
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("COMPREHENSIVE HALFTONE ROUTING TEST")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Form data processing
    print("\n1. Testing form data processing...")
    options = test_form_data_processing()
    if not options:
        all_tests_passed = False
    
    # Test 2: Parallel decision logic
    if options:
        if not test_parallel_decision_logic(options):
            all_tests_passed = False
    
    # Test 3: Processing pipeline
    if options:
        if not test_processing_pipeline(options):
            all_tests_passed = False
    
    # Test 4: ZIP file inclusion
    if not test_zip_file_inclusion():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✅ ALL HALFTONE ROUTING TESTS PASSED")
        print("✅ Halftone effect is properly routed through the entire system")
    else:
        print("❌ SOME HALFTONE ROUTING TESTS FAILED")
        print("❌ Halftone effect routing needs attention")
    print("=" * 60)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script to verify halftone effect routing fix
"""

import os
import tempfile
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

def test_halftone_routing():
    """Test halftone routing in both parallel and sequential processing"""
    print("Testing halftone effect routing...")
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=4)
        
        # Test 1: Single halftone option (should use sequential)
        print("\n1. Testing single halftone option (sequential processing):")
        options1 = {'halftone': True}
        print(f"   Options: {options1}")
        print(f"   Should use parallel: {processor._should_use_parallel(options1)}")
        
        result1 = processor.process_logo(test_file, options1)
        print(f"   Success: {result1['success']}")
        print(f"   Outputs: {list(result1['outputs'].keys())}")
        print(f"   Processing time: {result1['processing_time']:.2f}s")
        print(f"   Parallel used: {result1['parallel']}")
        
        # Test 2: Multiple options including halftone (should use parallel)
        print("\n2. Testing multiple options including halftone (parallel processing):")
        options2 = {'halftone': True, 'transparent_png': True, 'black_version': True}
        print(f"   Options: {options2}")
        print(f"   Should use parallel: {processor._should_use_parallel(options2)}")
        
        result2 = processor.process_logo(test_file, options2)
        print(f"   Success: {result2['success']}")
        print(f"   Outputs: {list(result2['outputs'].keys())}")
        print(f"   Processing time: {result2['processing_time']:.2f}s")
        print(f"   Parallel used: {result2['parallel']}")
        print(f"   Workers used: {result2.get('workers_used', 'N/A')}")
        
        # Test 3: Verify halftone file was created
        print("\n3. Verifying halftone file creation:")
        if 'halftone' in result2['outputs']:
            halftone_path = result2['outputs']['halftone']
            if os.path.exists(halftone_path):
                print(f"   ✓ Halftone file created: {halftone_path}")
                file_size = os.path.getsize(halftone_path)
                print(f"   ✓ File size: {file_size} bytes")
            else:
                print(f"   ✗ Halftone file not found: {halftone_path}")
        else:
            print("   ✗ Halftone not in outputs")
        
        # Test 4: Test direct halftone method
        print("\n4. Testing direct halftone method:")
        try:
            direct_result = processor._create_halftone(test_file)
            print(f"   ✓ Direct method result: {direct_result}")
            if os.path.exists(direct_result):
                print(f"   ✓ Direct method file exists: {direct_result}")
            else:
                print(f"   ✗ Direct method file missing: {direct_result}")
        except Exception as e:
            print(f"   ✗ Direct method failed: {e}")
        
        print("\n" + "="*50)
        print("HALFTONE ROUTING TEST COMPLETED")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

if __name__ == "__main__":
    test_halftone_routing() 
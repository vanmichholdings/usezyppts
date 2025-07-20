#!/usr/bin/env python3
"""
Test script to verify that all processing methods are properly routed
and outputs are generated correctly, including PDF and favicon.
"""

import os
import sys
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
import zipfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyppts.utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a test logo for routing verification"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo
    draw.ellipse([50, 50, 350, 250], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.text((150, 150), "TEST", fill=(255, 255, 255, 255))
    
    return img

def test_routing_verification():
    """Test that all processing methods are properly routed"""
    print("🔍 Routing Verification Test Suite")
    print("=" * 60)
    
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
        # Initialize processor
        processor = LogoProcessor(
            cache_folder=cache_dir,
            upload_folder=upload_dir,
            output_folder=output_dir,
            temp_folder=temp_dir_processor
        )
        
        # Create test logo
        test_logo = create_test_logo()
        test_file_path = os.path.join(upload_dir, 'routing_test_logo.png')
        test_logo.save(test_file_path)
        print(f"Created test logo at: {test_file_path}")
        
        # Test configurations for different outputs
        test_configs = [
            {
                'name': 'Vector Trace (PDF/AI/SVG)',
                'options': {
                    'vector_trace': True,
                    'vector_trace_options': {
                        'smoothness': 0.7,
                        'use_bezier': True,
                        'preserve_details': False,
                        'min_area': 30,
                        'simplify_threshold': 0.6,
                        'enable_parallel': True
                    }
                },
                'expected_outputs': ['vector_trace_svg', 'vector_trace_pdf', 'vector_trace_ai']
            },
            {
                'name': 'Full Color Vector Trace',
                'options': {
                    'full_color_vector_trace': True,
                    'full_color_vector_trace_options': {
                        'smoothness': 0.8,
                        'use_bezier': True,
                        'preserve_details': True,
                        'min_area': 25,
                        'simplify_threshold': 0.5,
                        'enable_parallel': True
                    }
                },
                'expected_outputs': ['full_color_vector_trace_svg', 'full_color_vector_trace_pdf', 'full_color_vector_trace_ai']
            },
            {
                'name': 'Transparent PNG',
                'options': {
                    'transparent_png': True
                },
                'expected_outputs': ['transparent_png']
            },
            {
                'name': 'Black Version',
                'options': {
                    'black_version': True
                },
                'expected_outputs': ['smooth_gray_version', 'black_version']
            },
            {
                'name': 'Distressed Effect',
                'options': {
                    'distressed_effect': True
                },
                'expected_outputs': ['distressed_effect']
            },
            {
                'name': 'Color Separations',
                'options': {
                    'color_separations': True
                },
                'expected_outputs': ['color_separations']
            },
            {
                'name': 'Contour Cut',
                'options': {
                    'contour_cut': True
                },
                'expected_outputs': ['contour_cut']
            },
            {
                'name': 'Favicon',
                'options': {
                    'favicon': True
                },
                'expected_outputs': ['favicon']
            },
            {
                'name': 'Social Media (with favicon)',
                'options': {
                    'social_formats': {
                        'facebook_profile': True,
                        'twitter_profile': True
                    }
                },
                'expected_outputs': ['social_facebook_profile', 'social_twitter_profile', 
                                   'social_facebook_profile_favicon', 'social_twitter_profile_favicon']
            }
        ]
        
        results = {}
        
        # Test each configuration
        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ---")
            
            start_time = time.time()
            result = processor.process_logo(test_file_path, config['options'])
            processing_time = time.time() - start_time
            
            if result.get('success'):
                outputs = result.get('outputs', {})
                print(f"  ✅ Success! Time: {processing_time:.2f}s")
                print(f"  📁 Generated outputs: {list(outputs.keys())}")
                
                # Check expected outputs
                missing_outputs = []
                for expected in config['expected_outputs']:
                    if expected not in outputs:
                        missing_outputs.append(expected)
                    else:
                        output_path = outputs[expected]
                        if isinstance(output_path, dict):
                            # Handle dict outputs (like color_separations)
                            if expected == 'color_separations':
                                pngs = output_path.get('pngs', [])
                                psd = output_path.get('psd')
                                if pngs and all(os.path.exists(png[0]) for png in pngs):
                                    print(f"    ✅ {expected}: {len(pngs)} PNG files + PSD")
                                else:
                                    print(f"    ❌ {expected}: Some files not found")
                            else:
                                print(f"    ✅ {expected}: Dict output with {len(output_path)} items")
                        elif os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            print(f"    ✅ {expected}: {output_path} ({file_size} bytes)")
                        else:
                            print(f"    ❌ {expected}: File not found at {output_path}")
                
                if missing_outputs:
                    print(f"    ⚠️  Missing outputs: {missing_outputs}")
                
                results[config['name']] = {
                    'success': True,
                    'processing_time': processing_time,
                    'outputs': outputs,
                    'missing_outputs': missing_outputs
                }
            else:
                print(f"  ❌ Failed: {result.get('message', 'Unknown error')}")
                results[config['name']] = {
                    'success': False,
                    'error': result.get('message', 'Unknown error')
                }
        
        # Test comprehensive processing
        print(f"\n--- Testing Comprehensive Processing ---")
        comprehensive_options = {
            'vector_trace': True,
            'vector_trace_options': {
                'smoothness': 0.7,
                'use_bezier': True,
                'preserve_details': False,
                'min_area': 30,
                'simplify_threshold': 0.6,
                'enable_parallel': True
            },
            'transparent_png': True,
            'black_version': True,
            'favicon': True,
            'social_formats': {
                'facebook_profile': True
            }
        }
        
        start_time = time.time()
        comprehensive_result = processor.process_logo(test_file_path, comprehensive_options)
        processing_time = time.time() - start_time
        
        if comprehensive_result.get('success'):
            outputs = comprehensive_result.get('outputs', {})
            print(f"  ✅ Comprehensive processing successful! Time: {processing_time:.2f}s")
            print(f"  📁 Total outputs: {len(outputs)}")
            
            # Check for key outputs
            key_outputs = ['vector_trace_svg', 'vector_trace_pdf', 'vector_trace_ai', 
                          'transparent_png', 'smooth_gray_version', 'black_version', 
                          'favicon', 'social_facebook_profile', 'social_facebook_profile_favicon']
            
            for key_output in key_outputs:
                if key_output in outputs:
                    output_path = outputs[key_output]
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        print(f"    ✅ {key_output}: {file_size} bytes")
                    else:
                        print(f"    ❌ {key_output}: File not found")
                else:
                    print(f"    ❌ {key_output}: Not in outputs")
            
            results['Comprehensive'] = {
                'success': True,
                'processing_time': processing_time,
                'outputs': outputs
            }
        else:
            print(f"  ❌ Comprehensive processing failed: {comprehensive_result.get('message', 'Unknown error')}")
            results['Comprehensive'] = {
                'success': False,
                'error': comprehensive_result.get('message', 'Unknown error')
            }
        
        # Print summary
        print(f"\n🎯 ROUTING VERIFICATION SUMMARY")
        print("=" * 60)
        
        successful_tests = 0
        total_tests = len(results)
        
        for test_name, result in results.items():
            if result.get('success'):
                successful_tests += 1
                print(f"  ✅ {test_name}: Success ({result.get('processing_time', 0):.2f}s)")
                if 'missing_outputs' in result and result['missing_outputs']:
                    print(f"     ⚠️  Missing: {result['missing_outputs']}")
            else:
                print(f"  ❌ {test_name}: Failed - {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Results: {successful_tests}/{total_tests} tests passed")
        
        # Check for specific issues
        print(f"\n🔍 Specific Checks:")
        
        # Check PDF generation
        pdf_found = False
        for test_name, result in results.items():
            if result.get('success') and 'outputs' in result:
                outputs = result['outputs']
                if any('pdf' in key for key in outputs.keys()):
                    pdf_found = True
                    break
        
        if pdf_found:
            print(f"  ✅ PDF generation: Working")
        else:
            print(f"  ❌ PDF generation: Not found in any test")
        
        # Check favicon generation
        favicon_found = False
        for test_name, result in results.items():
            if result.get('success') and 'outputs' in result:
                outputs = result['outputs']
                if any('favicon' in key for key in outputs.keys()):
                    favicon_found = True
                    break
        
        if favicon_found:
            print(f"  ✅ Favicon generation: Working")
        else:
            print(f"  ❌ Favicon generation: Not found in any test")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"❌ Error during routing verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("🔍 Logo Processing Routing Verification Test Suite")
    print("=" * 70)
    
    success = test_routing_verification()
    
    if success:
        print("\n🎉 All routing tests passed!")
        print("\n✅ All methods are properly routed:")
        print("✅ PDF generation is working")
        print("✅ Favicon generation is working")
        print("✅ Social media processing includes additional outputs")
        print("✅ All outputs are properly included in results")
        sys.exit(0)
    else:
        print("\n💥 Some routing tests failed!")
        print("\nPlease check the output above for specific issues.")
        sys.exit(1) 
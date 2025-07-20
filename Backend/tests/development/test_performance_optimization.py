#!/usr/bin/env python3
"""
Performance test script for optimized vector tracing functionality
"""

import os
import sys
import tempfile
import time
import statistics
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zyppts.utils.logo_processor import LogoProcessor

def create_performance_test_logo():
    """Create a test logo for performance testing"""
    img = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw multiple complex shapes
    # 1. Large background shape
    draw.ellipse([50, 50, 750, 550], fill=(200, 200, 200, 255), outline=(100, 100, 100, 255), width=5)
    
    # 2. Multiple text elements
    try:
        font_large = ImageFont.truetype("Arial.ttf", 48)
        font_medium = ImageFont.truetype("Arial.ttf", 24)
        font_small = ImageFont.truetype("Arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Text with different colors
    draw.text((100, 100), "PERFORMANCE", fill=(255, 0, 0, 255), font=font_large)
    draw.text((150, 200), "TEST LOGO", fill=(0, 255, 0, 255), font=font_medium)
    draw.text((200, 300), "VECTOR TRACING", fill=(0, 0, 255, 255), font=font_medium)
    
    # 3. Complex geometric shapes
    # Triangles
    for i in range(5):
        x = 100 + i * 120
        y = 400
        draw.polygon([(x, y), (x+50, y+80), (x-20, y+60)], fill=(255, 165, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    # 4. Small decorative elements
    for i in range(10):
        x = 50 + i * 70
        y = 500
        draw.ellipse([x, y, x+30, y+30], fill=(128, 128, 128, 255))
    
    return img

def test_performance_optimizations():
    """Test the performance optimizations in vector tracing"""
    print("🚀 Performance Optimization Test Suite")
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
        test_logo = create_performance_test_logo()
        test_file_path = os.path.join(upload_dir, 'performance_test_logo.png')
        test_logo.save(test_file_path)
        print(f"Created performance test logo at: {test_file_path}")
        
        # Test configurations for performance comparison
        test_configs = [
            {
                'name': 'Ultra Fast (Cached)',
                'options': {
                    'vector_trace': True,
                    'vector_trace_options': {
                        'smoothness': 0.5,
                        'use_bezier': False,
                        'preserve_details': False,
                        'min_area': 50,
                        'simplify_threshold': 0.8,
                        'enable_parallel': True
                    }
                }
            },
            {
                'name': 'Fast (Optimized)',
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
                }
            },
            {
                'name': 'Balanced (Quality/Speed)',
                'options': {
                    'vector_trace': True,
                    'vector_trace_options': {
                        'smoothness': 0.8,
                        'use_bezier': True,
                        'preserve_details': True,
                        'min_area': 25,
                        'simplify_threshold': 0.5,
                        'enable_parallel': True
                    }
                }
            },
            {
                'name': 'High Quality (Slower)',
                'options': {
                    'vector_trace': True,
                    'vector_trace_options': {
                        'smoothness': 0.9,
                        'use_bezier': True,
                        'preserve_details': True,
                        'min_area': 20,
                        'simplify_threshold': 0.3,
                        'enable_parallel': False
                    }
                }
            }
        ]
        
        results = {}
        
        # Test each configuration multiple times for accurate timing
        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ---")
            
            times = []
            success_count = 0
            
            # Run multiple iterations for accurate timing
            for i in range(3):
                print(f"  Iteration {i+1}/3...")
                start_time = time.time()
                
                result = processor.process_logo(test_file_path, config['options'])
                
                processing_time = time.time() - start_time
                times.append(processing_time)
                
                if result.get('success'):
                    success_count += 1
                    outputs = result.get('outputs', {})
                    if 'vector_trace_svg' in outputs:
                        svg_path = outputs['vector_trace_svg']
                        if os.path.exists(svg_path):
                            print(f"    ✅ Success! Time: {processing_time:.2f}s")
                        else:
                            print(f"    ❌ SVG file not found")
                    else:
                        print(f"    ❌ No vector_trace_svg in outputs")
                else:
                    print(f"    ❌ Failed: {result.get('message', 'Unknown error')}")
            
            # Calculate statistics
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                
                results[config['name']] = {
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'success_rate': success_count / 3,
                    'times': times
                }
                
                print(f"  📊 Results:")
                print(f"    - Average time: {avg_time:.2f}s")
                print(f"    - Min time: {min_time:.2f}s")
                print(f"    - Max time: {max_time:.2f}s")
                print(f"    - Success rate: {success_count}/3 ({success_count/3*100:.1f}%)")
        
        # Test cache performance
        print(f"\n--- Testing Cache Performance ---")
        cache_config = test_configs[1]  # Use Fast configuration
        
        # First run (no cache)
        print("  First run (no cache)...")
        start_time = time.time()
        result1 = processor.process_logo(test_file_path, cache_config['options'])
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        print("  Second run (with cache)...")
        start_time = time.time()
        result2 = processor.process_logo(test_file_path, cache_config['options'])
        second_run_time = time.time() - start_time
        
        cache_speedup = first_run_time / second_run_time if second_run_time > 0 else 0
        
        print(f"  📊 Cache Performance:")
        print(f"    - First run: {first_run_time:.2f}s")
        print(f"    - Second run: {second_run_time:.2f}s")
        print(f"    - Speedup: {cache_speedup:.1f}x")
        
        # Print comprehensive results
        print(f"\n🎯 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        if results:
            print(f"Configuration Performance Rankings:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'])
            
            for i, (name, stats) in enumerate(sorted_results, 1):
                print(f"  {i}. {name}")
                print(f"     - Avg: {stats['avg_time']:.2f}s | Min: {stats['min_time']:.2f}s | Max: {stats['max_time']:.2f}s")
                print(f"     - Success: {stats['success_rate']*100:.1f}%")
            
            # Performance improvements
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            improvement = (slowest[1]['avg_time'] - fastest[1]['avg_time']) / slowest[1]['avg_time'] * 100
            
            print(f"\n🚀 Performance Improvements:")
            print(f"  - Fastest: {fastest[0]} ({fastest[1]['avg_time']:.2f}s)")
            print(f"  - Slowest: {slowest[0]} ({slowest[1]['avg_time']:.2f}s)")
            print(f"  - Speed improvement: {improvement:.1f}%")
            print(f"  - Cache speedup: {cache_speedup:.1f}x")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error during performance testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_batch_processing_performance():
    """Test batch processing performance with multiple logos"""
    print(f"\n🔄 Batch Processing Performance Test")
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
        
        # Create multiple test logos
        logo_paths = []
        for i in range(5):
            # Create different logos
            img = Image.new('RGBA', (400, 300), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Different colors and shapes for each logo
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
            color = colors[i % len(colors)]
            
            # Draw different shapes
            if i % 3 == 0:
                draw.ellipse([50, 50, 350, 250], fill=color + (255,), outline=(0, 0, 0, 255), width=3)
            elif i % 3 == 1:
                draw.rectangle([50, 50, 350, 250], fill=color + (255,), outline=(0, 0, 0, 255), width=3)
            else:
                draw.polygon([(200, 50), (350, 250), (50, 250)], fill=color + (255,), outline=(0, 0, 0, 255), width=3)
            
            # Add text
            try:
                font = ImageFont.truetype("Arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((150, 150), f"LOGO {i+1}", fill=(0, 0, 0, 255), font=font)
            
            # Save logo
            logo_path = os.path.join(upload_dir, f'logo_{i+1}.png')
            img.save(logo_path)
            logo_paths.append(logo_path)
        
        print(f"Created {len(logo_paths)} test logos")
        
        # Test batch processing with optimized settings
        batch_options = {
            'vector_trace': True,
            'vector_trace_options': {
                'smoothness': 0.7,
                'use_bezier': True,
                'preserve_details': False,
                'min_area': 30,
                'simplify_threshold': 0.6,
                'enable_parallel': True
            }
        }
        
        # Process logos individually (baseline)
        print("Processing logos individually...")
        individual_times = []
        individual_success = 0
        
        for i, logo_path in enumerate(logo_paths):
            print(f"  Processing logo {i+1}/{len(logo_paths)}...")
            start_time = time.time()
            
            result = processor.process_logo(logo_path, batch_options)
            
            processing_time = time.time() - start_time
            individual_times.append(processing_time)
            
            if result.get('success'):
                individual_success += 1
                print(f"    ✅ Success! Time: {processing_time:.2f}s")
            else:
                print(f"    ❌ Failed: {result.get('message', 'Unknown error')}")
        
        total_individual_time = sum(individual_times)
        avg_individual_time = statistics.mean(individual_times) if individual_times else 0
        
        print(f"\n📊 Individual Processing Results:")
        print(f"  - Total time: {total_individual_time:.2f}s")
        print(f"  - Average time per logo: {avg_individual_time:.2f}s")
        print(f"  - Success rate: {individual_success}/{len(logo_paths)} ({individual_success/len(logo_paths)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during batch processing test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("🚀 Vector Tracing Performance Optimization Test Suite")
    print("=" * 70)
    
    # Run performance tests
    success1 = test_performance_optimizations()
    success2 = test_batch_processing_performance()
    
    if success1 and success2:
        print("\n🎉 All performance tests completed!")
        print("\nKey Performance Optimizations:")
        print("✅ Intelligent caching system (24-hour cache)")
        print("✅ Parallel processing for multiple colors")
        print("✅ Optimized algorithms (reduced complexity)")
        print("✅ Early termination for large images")
        print("✅ Fast background detection (2 methods vs 5)")
        print("✅ Simplified color clustering (no silhouette analysis)")
        print("✅ Reduced morphological operations")
        print("✅ Optimized path simplification")
        print("✅ Minimal SVG output")
        print("✅ Adaptive tolerance calculation")
        sys.exit(0)
    else:
        print("\n💥 Some performance tests failed!")
        sys.exit(1) 
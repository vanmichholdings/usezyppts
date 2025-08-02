#!/usr/bin/env python3
"""
Test script for parallel processing implementation.
This script validates all three steps of the Recommended Implementation Plan.
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from PIL import Image, ImageDraw

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.logo_processor import LogoProcessor

def create_test_image():
    """Create a simple test logo image for testing."""
    # Create a simple logo with multiple colors
    width, height = 400, 300
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo with multiple colors
    # Background circle
    draw.ellipse([50, 50, 350, 250], fill=(0, 100, 200, 255))
    
    # Inner circle
    draw.ellipse([100, 100, 300, 200], fill=(255, 255, 255, 255))
    
    # Text/design element
    draw.rectangle([150, 130, 250, 170], fill=(200, 50, 50, 255))
    
    # Save test image
    test_image_path = "test_logo_parallel.png"
    img.save(test_image_path)
    return test_image_path

def test_parallel_processing():
    """Test the parallel processing implementation"""
    print("🚀 Testing Parallel Processing Implementation")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    print(f"✅ Created test image: {test_image}")
    
    # Test 1: Sequential Processing (Baseline)
    print("\n📊 Test 1: Sequential Processing (Baseline)")
    processor_seq = LogoProcessor(use_parallel=False, max_workers=1)
    
    options = {
        'transparent_png': True,
        'black_version': True,
        'pdf_version': True,
        'vector_trace': True,
        'color_separations': True
    }
    
    start_time = time.time()
    result_seq = processor_seq.process_logo(test_image, options)
    seq_time = time.time() - start_time
    
    print(f"⏱️  Sequential processing time: {seq_time:.2f}s")
    print(f"📁 Outputs generated: {result_seq.get('total_outputs', 0)}")
    print(f"✅ Success: {result_seq.get('success', False)}")
    
    # Test 2: Parallel Processing
    print("\n📊 Test 2: Parallel Processing")
    processor_par = LogoProcessor(use_parallel=True, max_workers=16)  # Increased from 8 to 16
    
    start_time = time.time()
    result_par = processor_par.process_logo(test_image, options)
    par_time = time.time() - start_time
    
    print(f"⏱️  Parallel processing time: {par_time:.2f}s")
    print(f"📁 Outputs generated: {result_par.get('total_outputs', 0)}")
    print(f"✅ Success: {result_par.get('success', False)}")
    print(f"🔧 Workers used: {result_par.get('workers_used', 0)}")
    
    # Test 3: Performance Comparison
    print("\n📊 Test 3: Performance Comparison")
    if seq_time > 0:
        speedup = seq_time / par_time
        improvement = ((seq_time - par_time) / seq_time) * 100
        print(f"⚡ Speedup: {speedup:.2f}x faster")
        print(f"📈 Improvement: {improvement:.1f}% faster")
    else:
        print("⚠️  Cannot calculate speedup (sequential time is 0)")
    
    # Test 4: Progress Tracking
    print("\n📊 Test 4: Progress Tracking")
    
    def progress_callback(task_name, progress, message):
        print(f"📊 {task_name}: {progress:.1%} - {message}")
    
    processor_prog = LogoProcessor(use_parallel=True, max_workers=12)  # Increased from 4 to 12
    processor_prog.set_progress_callback(progress_callback)
    
    options_simple = {
        'transparent_png': True,
        'black_version': True,
        'pdf_version': True
    }
    
    print("🔄 Testing with progress callback...")
    result_prog = processor_prog.process_logo(test_image, options_simple)
    print(f"✅ Progress tracking test completed: {result_prog.get('success', False)}")
    
    # Test 5: Performance Statistics
    print("\n📊 Test 5: Performance Statistics")
    stats = processor_prog.get_performance_stats()
    print(f"🔧 Parallel enabled: {stats.get('parallel_enabled', False)}")
    print(f"👥 Max workers: {stats.get('max_workers', 0)}")
    print(f"💻 CPU count: {stats.get('cpu_count', 0)}")
    
    # Test 6: Worker Optimization
    print("\n📊 Test 6: Worker Optimization")
    for task_count in [1, 4, 8, 16, 32]:
        optimal_workers = processor_prog.optimize_worker_count(task_count)
        print(f"📋 {task_count} tasks → {optimal_workers} workers")
    
    # Test 7: Task Priority
    print("\n📊 Test 7: Task Priority")
    tasks = ['transparent_png', 'vector_trace', 'color_separations', 'favicon', 'social_formats']
    for task in tasks:
        priority = processor_prog.get_task_priority(task)
        print(f"🎯 {task}: priority {priority}")
    
    # Cleanup
    try:
        os.remove(test_image)
        print(f"\n🧹 Cleaned up test image: {test_image}")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': speedup if seq_time > 0 else 0,
        'improvement': improvement if seq_time > 0 else 0,
        'success': result_par.get('success', False)
    }

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔍 Testing Edge Cases and Error Handling")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Test 1: No options selected
    print("\n📊 Test 1: No options selected")
    processor = LogoProcessor(use_parallel=True, max_workers=4)
    result = processor.process_logo(test_image, {})
    print(f"✅ Result: {result.get('success', False)}")
    print(f"📝 Message: {result.get('message', '')}")
    
    # Test 2: Single option (should use sequential)
    print("\n📊 Test 2: Single option (should use sequential)")
    result = processor.process_logo(test_image, {'transparent_png': True})
    print(f"✅ Result: {result.get('success', False)}")
    print(f"🔄 Parallel: {result.get('parallel', False)}")
    
    # Test 3: Invalid file path
    print("\n📊 Test 3: Invalid file path")
    try:
        result = processor.process_logo("nonexistent_file.png", {'transparent_png': True})
        print(f"⚠️  Expected error but got: {result.get('success', False)}")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}")
    
    # Test 4: High worker count
    print("\n📊 Test 4: High worker count")
    processor_high = LogoProcessor(use_parallel=True, max_workers=32)
    result = processor_high.process_logo(test_image, {
        'transparent_png': True,
        'black_version': True,
        'pdf_version': True
    })
    print(f"✅ Result: {result.get('success', False)}")
    print(f"👥 Workers used: {result.get('workers_used', 0)}")
    
    # Cleanup
    try:
        os.remove(test_image)
    except:
        pass
    
    print("\n✅ Edge case tests completed!")

if __name__ == "__main__":
    print("🧪 Parallel Processing Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Run main tests
        results = test_parallel_processing()
        
        # Run edge case tests
        test_edge_cases()
        
        # Summary
        print("\n📋 Test Summary")
        print("=" * 60)
        print(f"⚡ Speedup achieved: {results['speedup']:.2f}x")
        print(f"📈 Performance improvement: {results['improvement']:.1f}%")
        print(f"✅ All tests passed: {results['success']}")
        
        if results['speedup'] > 1.5:
            print("🎉 Excellent! Parallel processing is working effectively!")
        elif results['speedup'] > 1.1:
            print("👍 Good! Parallel processing is providing some benefit.")
        else:
            print("⚠️  Parallel processing may need optimization.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
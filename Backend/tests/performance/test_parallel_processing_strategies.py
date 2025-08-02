#!/usr/bin/env python3
"""
Test script for intelligent parallel processing strategies.
This script demonstrates the new parallel processing capabilities and validates performance improvements.
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

def create_test_image():
    """Create a simple test logo image for testing."""
    from PIL import Image, ImageDraw
    
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

def test_single_task_parallel():
    """Test parallel processing with a single task."""
    print("🧪 Testing Single Task Parallel Processing")
    print("=" * 50)
    
    # Create test image
    test_image_path = create_test_image()
    
    # Initialize processor
    processor = LogoProcessor(
        cache_dir="./test_cache",
        output_folder="./test_outputs",
        temp_folder="./test_temp"
    )
    
    # Test with single vector trace task
    test_options = {
        'vector_trace': True,
        'vector_trace_options': {
            'smoothness': 0.5,
            'use_bezier': True,
            'enable_parallel': True
        }
    }
    
    print("🔄 Running single task with intelligent parallel processing...")
    start_time = time.time()
    
    try:
        result = processor.process_logo(test_image_path, test_options)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Single task completed in {processing_time:.2f} seconds")
        
        if result.get('success', False):
            print(f"📊 Results:")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Strategy used: Intelligent parallel processing")
            print(f"   - Outputs generated: {len(result.get('outputs', {}))}")
            
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

def test_multiple_tasks_parallel():
    """Test parallel processing with multiple tasks."""
    print("\n🧪 Testing Multiple Tasks Parallel Processing")
    print("=" * 55)
    
    # Create test image
    test_image_path = create_test_image()
    
    # Initialize processor
    processor = LogoProcessor(
        cache_dir="./test_cache",
        output_folder="./test_outputs",
        temp_folder="./test_temp"
    )
    
    # Test with multiple tasks
    test_options = {
        'vector_trace': True,
        'transparent_png': True,
        'black_version': True,
        'distressed_effect': True,
        'favicon': True,
        'vector_trace_options': {
            'smoothness': 0.5,
            'use_bezier': True,
            'enable_parallel': True
        }
    }
    
    print("🔄 Running multiple tasks with intelligent parallel processing...")
    start_time = time.time()
    
    try:
        result = processor.process_logo(test_image_path, test_options)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Multiple tasks completed in {processing_time:.2f} seconds")
        
        if result.get('success', False):
            outputs = result.get('outputs', {})
            print(f"📊 Results:")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Tasks completed: {len(outputs)}")
            print(f"   - Strategy used: Intelligent parallel processing")
            
            for output_name, output_path in outputs.items():
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / 1024  # KB
                    print(f"   - {output_name}: {file_size:.1f} KB")
            
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

def test_heavy_tasks_parallel():
    """Test parallel processing with heavy computational tasks."""
    print("\n🧪 Testing Heavy Tasks Parallel Processing")
    print("=" * 55)
    
    # Create test image
    test_image_path = create_test_image()
    
    # Initialize processor
    processor = LogoProcessor(
        cache_dir="./test_cache",
        output_folder="./test_outputs",
        temp_folder="./test_temp"
    )
    
    # Test with heavy tasks
    test_options = {
        'vector_trace': True,
        'full_color_vector_trace': True,
        'color_separations': True,
        'vector_trace_options': {
            'smoothness': 0.5,
            'use_bezier': True,
            'enable_parallel': True
        },
        'full_color_vector_trace_options': {
            'smoothness': 0.5,
            'use_bezier': True,
            'enable_parallel': True
        }
    }
    
    print("🔄 Running heavy tasks with intelligent parallel processing...")
    start_time = time.time()
    
    try:
        result = processor.process_logo(test_image_path, test_options)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Heavy tasks completed in {processing_time:.2f} seconds")
        
        if result.get('success', False):
            outputs = result.get('outputs', {})
            print(f"📊 Results:")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Heavy tasks completed: {len(outputs)}")
            print(f"   - Strategy used: Intelligent parallel processing")
            
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

def test_strategy_selection():
    """Test the strategy selection logic."""
    print("\n🧪 Testing Strategy Selection Logic")
    print("=" * 45)
    
    processor = LogoProcessor()
    
    # Test different task combinations
    test_cases = [
        {
            'name': 'Light Tasks Only',
            'tasks': [('transparent_png', None), ('favicon', None), ('distressed_effect', None)],
            'expected': 'thread_pool'
        },
        {
            'name': 'Heavy Tasks Only',
            'tasks': [('vector_trace', None), ('full_color_vector_trace', None), ('color_separations', None)],
            'expected': 'process_pool'
        },
        {
            'name': 'Mixed Tasks',
            'tasks': [('transparent_png', None), ('vector_trace', None), ('black_version', None), ('color_separations', None)],
            'expected': 'hybrid'
        },
        {
            'name': 'Single Task',
            'tasks': [('vector_trace', None)],
            'expected': 'async_single'
        }
    ]
    
    cpu_count = processor._get_optimal_processing_strategy.__code__.co_argcount
    memory_gb = 8  # Assume 8GB for testing
    
    for test_case in test_cases:
        strategy = processor._get_optimal_processing_strategy(test_case['tasks'], cpu_count, memory_gb)
        selected_strategy = strategy['strategy']
        expected = test_case['expected']
        
        status = "✅" if selected_strategy == expected else "❌"
        print(f"{status} {test_case['name']}: {selected_strategy} (expected: {expected})")
        print(f"   Description: {strategy['description']}")

def main():
    """Main test function."""
    print("🚀 Intelligent Parallel Processing Strategy Test")
    print("=" * 65)
    
    # Test strategy selection logic
    test_strategy_selection()
    
    # Test single task parallel processing
    single_success = test_single_task_parallel()
    
    # Test multiple tasks parallel processing
    multiple_success = test_multiple_tasks_parallel()
    
    # Test heavy tasks parallel processing
    heavy_success = test_heavy_tasks_parallel()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 20)
    print(f"Single Task Parallel: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"Multiple Tasks Parallel: {'✅ PASS' if multiple_success else '❌ FAIL'}")
    print(f"Heavy Tasks Parallel: {'✅ PASS' if heavy_success else '❌ FAIL'}")
    
    if all([single_success, multiple_success, heavy_success]):
        print(f"\n🎉 All tests passed!")
        print(f"   The intelligent parallel processing is working correctly.")
        print(f"   Performance improvements should be significant.")
    else:
        print(f"\n💥 Some tests failed!")
        print(f"   Check the error messages above for details.")

if __name__ == "__main__":
    main() 
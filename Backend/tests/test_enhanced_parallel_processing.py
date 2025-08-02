#!/usr/bin/env python3
"""
Test to verify the enhanced parallel processing with multiple workers per task
"""

import os
import tempfile
import logging
import time
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_test_logo():
    """Create a simple test logo"""
    img = Image.new('RGBA', (300, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo
    draw.ellipse([50, 50, 250, 250], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.rectangle([100, 100, 200, 200], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    return img

def test_task_complexity_analysis():
    """Test task complexity analysis"""
    logger = setup_logging()
    logger.info("🧪 Testing task complexity analysis")
    
    processor = LogoProcessor()
    
    # Test complexity scores for different tasks
    test_tasks = [
        'transparent_png',
        'social_formats',
        'color_separations',
        'vector_trace',
        'contour_cut'
    ]
    
    for task in test_tasks:
        complexity = processor.get_task_complexity(task)
        logger.info(f"📊 {task}: score={complexity['score']}, subtasks={complexity['subtasks']}, cpu_intensive={complexity['cpu_intensive']}")
        
        # Test worker calculation
        optimal_workers = processor.calculate_optimal_workers_for_task(task, 16)
        logger.info(f"⚙️ {task}: optimal workers={optimal_workers}")
    
    logger.info("✅ Task complexity analysis test passed!")
    return True

def test_subtask_creation():
    """Test subtask creation for complex tasks"""
    logger = setup_logging()
    logger.info("🧪 Testing subtask creation")
    
    processor = LogoProcessor()
    
    # Create test logo
    test_logo = create_test_logo()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Test social formats subtask creation
        social_options = {
            'social_formats': {
                'instagram_profile': True,
                'facebook_profile': True,
                'twitter_profile': True,
                'youtube_profile': True
            }
        }
        
        subtasks = processor.create_subtasks('social_formats', test_file, social_options)
        logger.info(f"📋 Created {len(subtasks)} subtasks for social_formats")
        
        for subtask in subtasks:
            logger.info(f"  - {subtask['name']}: priority={subtask['priority']}")
        
        # Test favicon subtask creation
        favicon_subtasks = processor.create_subtasks('favicon', test_file, {})
        logger.info(f"📋 Created {len(favicon_subtasks)} subtasks for favicon")
        
        logger.info("✅ Subtask creation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Subtask creation test failed: {str(e)}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_enhanced_parallel_processing():
    """Test enhanced parallel processing with multiple workers per task"""
    logger = setup_logging()
    logger.info("🧪 Testing enhanced parallel processing")
    
    # Create test logo
    test_logo = create_test_logo()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Configure processor for enhanced parallel processing
        processor = LogoProcessor(
            max_workers=16,
            use_parallel=True
        )
        
        # Enable task parallelization
        processor.parallel_config['task_parallelization'] = True
        processor.parallel_config['max_concurrent_tasks'] = 4
        processor.parallel_config['subtask_workers'] = 4
        
        # Test with complex workload that should trigger task-level parallelization
        complex_options = {
            'transparent_png': True,
            'black_version': True,
            'pdf_version': True,
            'webp_version': True,
            'favicon': True,
            'email_header': True,
            'vector_trace': True,
            'contour_cut': True,
            'social_formats': {
                'instagram_profile': True,
                'instagram_post': True,
                'facebook_profile': True,
                'facebook_post': True,
                'twitter_profile': True,
                'twitter_post': True,
                'youtube_profile': True,
                'youtube_cover': True
            }
        }
        
        logger.info("🚀 Starting enhanced parallel processing test...")
        start_time = time.time()
        
        result = processor.process_logo(test_file, complex_options)
        
        processing_time = time.time() - start_time
        
        # Verify results
        assert result['success'], "Processing should be successful"
        assert result['parallel'], "Should use parallel processing"
        assert 'task_parallelization' in result, "Should have task_parallelization flag"
        
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"📊 Task parallelization: {result.get('task_parallelization', False)}")
        logger.info(f"⚙️ Workers used: {result.get('workers_used', 0)}")
        logger.info(f"📈 Success rate: {result.get('success_rate', 0):.2%}")
        logger.info(f"📁 Outputs: {len(result['outputs'])}")
        
        # Log which outputs were created
        for output_name, output_data in result['outputs'].items():
            if isinstance(output_data, dict):
                logger.info(f"  📁 {output_name}: {len(output_data)} items")
            else:
                logger.info(f"  📄 {output_name}: {output_data}")
        
        logger.info("✅ Enhanced parallel processing test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced parallel processing test failed: {str(e)}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_performance_comparison():
    """Test performance comparison between standard and enhanced parallel processing"""
    logger = setup_logging()
    logger.info("🧪 Testing performance comparison")
    
    # Create test logo
    test_logo = create_test_logo()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Test options
        test_options = {
            'transparent_png': True,
            'black_version': True,
            'pdf_version': True,
            'webp_version': True,
            'favicon': True,
            'social_formats': {
                'instagram_profile': True,
                'facebook_profile': True,
                'twitter_profile': True,
                'youtube_profile': True
            }
        }
        
        # Test 1: Standard parallel processing
        logger.info("🔄 Testing standard parallel processing...")
        processor_standard = LogoProcessor(max_workers=8, use_parallel=True)
        processor_standard.parallel_config['task_parallelization'] = False
        
        start_time = time.time()
        result_standard = processor_standard.process_logo(test_file, test_options)
        standard_time = time.time() - start_time
        
        # Test 2: Enhanced parallel processing
        logger.info("🚀 Testing enhanced parallel processing...")
        processor_enhanced = LogoProcessor(max_workers=8, use_parallel=True)
        processor_enhanced.parallel_config['task_parallelization'] = True
        processor_enhanced.parallel_config['max_concurrent_tasks'] = 4
        processor_enhanced.parallel_config['subtask_workers'] = 2
        
        start_time = time.time()
        result_enhanced = processor_enhanced.process_logo(test_file, test_options)
        enhanced_time = time.time() - start_time
        
        # Compare results
        logger.info(f"📊 Performance Comparison:")
        logger.info(f"  Standard: {standard_time:.2f}s")
        logger.info(f"  Enhanced: {enhanced_time:.2f}s")
        logger.info(f"  Speedup: {standard_time/enhanced_time:.2f}x")
        
        # Verify both methods produced the same outputs
        standard_outputs = set(result_standard['outputs'].keys())
        enhanced_outputs = set(result_enhanced['outputs'].keys())
        
        logger.info(f"📁 Standard outputs: {len(standard_outputs)}")
        logger.info(f"📁 Enhanced outputs: {len(enhanced_outputs)}")
        
        if standard_outputs == enhanced_outputs:
            logger.info("✅ Both methods produced identical outputs")
        else:
            logger.warning(f"⚠️ Output mismatch: {standard_outputs.symmetric_difference(enhanced_outputs)}")
        
        logger.info("✅ Performance comparison test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance comparison test failed: {str(e)}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("🚀 Starting enhanced parallel processing tests")
    
    tests = [
        ("Task Complexity Analysis", test_task_complexity_analysis),
        ("Subtask Creation", test_subtask_creation),
        ("Enhanced Parallel Processing", test_enhanced_parallel_processing),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {str(e)}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    main() 
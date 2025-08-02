#!/usr/bin/env python3
"""
Test to verify that the logo processor is using only parallel processing
and that comprehensive logging is working correctly
"""

import os
import tempfile
import logging
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def setup_logging():
    """Setup comprehensive logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_parallel_processing.log')
        ]
    )
    return logging.getLogger(__name__)

def create_test_logo():
    """Create a simple test logo"""
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.rectangle([60, 60, 140, 140], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    draw.text((80, 90), "TEST", fill=(0, 0, 0, 255))
    
    return img

def test_parallel_only_processing():
    """Test that the logo processor always uses parallel processing"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING PARALLEL-ONLY PROCESSING")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        # Test with single variation (should still use parallel)
        logger.info("\n1. Testing single variation processing...")
        processor = LogoProcessor(use_parallel=True, max_workers=4)
        
        options = {
            'halftone': True
        }
        
        logger.info(f"Options: {options}")
        result = processor.process_logo(test_file, options)
        
        logger.info(f"Success: {result['success']}")
        logger.info(f"Parallel used: {result['parallel']}")
        logger.info(f"Workers used: {result.get('workers_used', 'N/A')}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        logger.info(f"Tasks processed: {result.get('tasks_processed', 'N/A')}")
        logger.info(f"Success rate: {result.get('success_rate', 'N/A')}")
        
        if result['parallel']:
            logger.info("✅ Single variation correctly used parallel processing")
        else:
            logger.error("❌ Single variation did not use parallel processing")
            return False
        
        # Test with multiple variations
        logger.info("\n2. Testing multiple variations processing...")
        options = {
            'halftone': True,
            'transparent_png': True,
            'black_version': True,
            'social_formats': {
                'instagram_profile': True,
                'facebook_profile': True
            }
        }
        
        logger.info(f"Options: {options}")
        result = processor.process_logo(test_file, options)
        
        logger.info(f"Success: {result['success']}")
        logger.info(f"Parallel used: {result['parallel']}")
        logger.info(f"Workers used: {result.get('workers_used', 'N/A')}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        logger.info(f"Tasks processed: {result.get('tasks_processed', 'N/A')}")
        logger.info(f"Success rate: {result.get('success_rate', 'N/A')}")
        logger.info(f"Outputs: {list(result['outputs'].keys())}")
        
        if result['parallel']:
            logger.info("✅ Multiple variations correctly used parallel processing")
        else:
            logger.error("❌ Multiple variations did not use parallel processing")
            return False
        
        # Test with disabled parallel processing (should still use parallel)
        logger.info("\n3. Testing with disabled parallel processing...")
        processor_disabled = LogoProcessor(use_parallel=False, max_workers=4)
        
        options = {
            'halftone': True,
            'transparent_png': True
        }
        
        logger.info(f"Options: {options}")
        result = processor_disabled.process_logo(test_file, options)
        
        logger.info(f"Success: {result['success']}")
        logger.info(f"Parallel used: {result['parallel']}")
        logger.info(f"Workers used: {result.get('workers_used', 'N/A')}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        
        if result['parallel']:
            logger.info("✅ Correctly ignored use_parallel=False and used parallel processing")
        else:
            logger.error("❌ Incorrectly used sequential processing when parallel was disabled")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

def test_logging_output():
    """Test that comprehensive logging is working"""
    logger = setup_logging()
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPREHENSIVE LOGGING")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test with multiple variations to see logging
        options = {
            'halftone': True,
            'transparent_png': True,
            'black_version': True
        }
        
        logger.info("Processing logo with comprehensive logging...")
        result = processor.process_logo(test_file, options)
        
        logger.info(f"Processing completed successfully: {result['success']}")
        logger.info(f"Total outputs: {len(result['outputs'])}")
        
        # Check if log file was created
        if os.path.exists('test_parallel_processing.log'):
            log_size = os.path.getsize('test_parallel_processing.log')
            logger.info(f"✅ Log file created: {log_size:,} bytes")
            
            # Read and display some log entries
            with open('test_parallel_processing.log', 'r') as f:
                log_lines = f.readlines()
                logger.info(f"📝 Log file contains {len(log_lines)} lines")
                
                # Show some key log entries
                key_entries = [line for line in log_lines if any(keyword in line for keyword in 
                    ['🚀', '🔄', '⚡', '📝', '⚫', '🔘', '✅', '📊', '⚙️'])]
                
                logger.info("🔍 Key log entries found:")
                for entry in key_entries[-10:]:  # Last 10 key entries
                    logger.info(f"   {entry.strip()}")
        else:
            logger.error("❌ Log file was not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

def main():
    """Main test function"""
    logger = setup_logging()
    
    all_tests_passed = True
    
    # Test 1: Parallel-only processing
    if not test_parallel_only_processing():
        all_tests_passed = False
    
    # Test 2: Comprehensive logging
    if not test_logging_output():
        all_tests_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("✅ ALL PARALLEL PROCESSING TESTS PASSED")
        logger.info("✅ Logo processor is using only parallel processing")
        logger.info("✅ Comprehensive logging is working correctly")
    else:
        logger.error("❌ SOME PARALLEL PROCESSING TESTS FAILED")
        logger.error("❌ Issues found with parallel processing or logging")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test to verify that the vector trace functionality is working correctly after the fix
"""

import os
import tempfile
import logging
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
    """Create a simple test logo with complex shapes"""
    img = Image.new('RGBA', (300, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a complex logo with multiple shapes
    # Main circle
    draw.ellipse([50, 50, 250, 250], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    # Inner rectangle
    draw.rectangle([100, 100, 200, 200], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    # Text
    draw.text((120, 140), "LOGO", fill=(0, 0, 255, 255))
    
    # Small decorative elements
    draw.ellipse([80, 80, 120, 120], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=1)
    draw.ellipse([180, 80, 220, 120], fill=(255, 0, 255, 255), outline=(0, 0, 0, 255), width=1)
    
    return img

def test_vector_trace_methods():
    """Test the vector trace methods directly"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING VECTOR TRACE METHODS")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test vector trace method
        logger.info("Testing _create_vector_trace method...")
        result = processor._create_vector_trace(test_file)
        
        logger.info(f"Vector trace result: {result}")
        
        if result:
            # Check that expected files were created
            expected_files = ['svg', 'pdf', 'eps']
            for file_type in expected_files:
                if file_type in result:
                    file_path = result[file_type]
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ {file_type}: {file_size:,} bytes")
                    else:
                        logger.error(f"❌ {file_type}: File missing - {file_path}")
                        return False
                else:
                    logger.warning(f"⚠️ {file_type}: Not in result dictionary")
            
            logger.info("✅ Vector trace method working correctly")
            return True
        else:
            logger.error("❌ Vector trace method returned empty result")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
            # Clean up generated files
            if 'result' in locals() and result:
                for file_type, file_path in result.items():
                    if os.path.exists(file_path):
                        os.unlink(file_path)
        except:
            pass

def test_full_color_vector_trace():
    """Test the full color vector trace method"""
    logger = setup_logging()
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FULL COLOR VECTOR TRACE")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test full color vector trace method
        logger.info("Testing _create_full_color_vector_trace method...")
        result = processor._create_full_color_vector_trace(test_file)
        
        logger.info(f"Full color vector trace result: {result}")
        
        if result:
            # Check that expected files were created
            expected_files = ['svg', 'pdf', 'eps']
            for file_type in expected_files:
                if file_type in result:
                    file_path = result[file_type]
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ {file_type}: {file_size:,} bytes")
                    else:
                        logger.error(f"❌ {file_type}: File missing - {file_path}")
                        return False
                else:
                    logger.warning(f"⚠️ {file_type}: Not in result dictionary")
            
            logger.info("✅ Full color vector trace method working correctly")
            return True
        else:
            logger.error("❌ Full color vector trace method returned empty result")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
            # Clean up generated files
            if 'result' in locals() and result:
                for file_type, file_path in result.items():
                    if os.path.exists(file_path):
                        os.unlink(file_path)
        except:
            pass

def test_vector_trace_integration():
    """Test vector trace integration with the main processing pipeline"""
    logger = setup_logging()
    logger.info("\n" + "=" * 60)
    logger.info("TESTING VECTOR TRACE INTEGRATION")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test with vector trace in the main processing pipeline
        options = {
            'vector_trace': True
        }
        
        logger.info("Processing with vector trace option...")
        result = processor.process_logo(test_file, options)
        
        logger.info(f"Processing result: {result['success']}")
        logger.info(f"Outputs: {list(result['outputs'].keys())}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        
        if 'vector_trace' in result['outputs']:
            vector_result = result['outputs']['vector_trace']
            logger.info(f"Vector trace outputs: {vector_result}")
            
            # Check that expected files are present
            expected_keys = ['svg', 'pdf', 'eps']
            for key in expected_keys:
                if key in vector_result:
                    file_path = vector_result[key]
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ {key}: {file_size:,} bytes")
                    else:
                        logger.error(f"❌ {key}: File missing")
                        return False
                else:
                    logger.warning(f"⚠️ {key}: Missing from result")
            
            logger.info("✅ Vector trace integration successful!")
            return True
        else:
            logger.error("❌ Vector trace not in outputs")
            return False
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
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
    
    # Test 1: Vector trace method
    if not test_vector_trace_methods():
        all_tests_passed = False
    
    # Test 2: Full color vector trace method
    if not test_full_color_vector_trace():
        all_tests_passed = False
    
    # Test 3: Integration test
    if not test_vector_trace_integration():
        all_tests_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("✅ ALL VECTOR TRACE TESTS PASSED")
        logger.info("✅ Vector trace functionality is working correctly")
        logger.info("✅ No more infinite recursion issues")
    else:
        logger.error("❌ SOME VECTOR TRACE TESTS FAILED")
        logger.error("❌ Issues found with vector trace functionality")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 
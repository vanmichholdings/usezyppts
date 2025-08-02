#!/usr/bin/env python3
"""
Test to verify the enhanced contour cutline functionality with better edge detection
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

def create_test_logo_with_nested_contours():
    """Create a test logo with nested contours (like letters with holes)"""
    img = Image.new('RGBA', (400, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a complex logo with nested contours
    # Main outer rectangle
    draw.rectangle([50, 50, 350, 250], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    
    # Inner rectangle (nested contour)
    draw.rectangle([100, 100, 300, 200], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    # Circle with hole (nested contour)
    draw.ellipse([150, 120, 250, 220], fill=(0, 0, 255, 255), outline=(0, 0, 0, 255), width=2)
    # Inner circle (hole)
    draw.ellipse([170, 140, 230, 200], fill=(255, 255, 255, 0), outline=(255, 255, 255, 255), width=1)
    
    # Text-like shape with holes
    # Letter "A" shape
    draw.polygon([(200, 80), (220, 120), (240, 80), (235, 90), (205, 90)], 
                fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    # Inner triangle (hole in A)
    draw.polygon([(210, 95), (220, 115), (230, 95)], 
                fill=(255, 255, 255, 0), outline=(255, 255, 255, 255), width=1)
    
    return img

def test_enhanced_contour_cutline():
    """Test the enhanced contour cutline functionality"""
    logger = setup_logging()
    logger.info("🧪 Testing enhanced contour cutline functionality")
    
    # Create test logo
    test_logo = create_test_logo_with_nested_contours()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Initialize processor
        processor = LogoProcessor()
        
        # Test contour cutline
        logger.info("🔍 Testing enhanced contour cutline...")
        result = processor._create_contour_cutline(test_file)
        
        # Verify results
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'full_color' in result, "Should have full color mask"
        assert 'pink_outline' in result, "Should have pink outline mask"
        assert 'svg' in result, "Should have combined SVG"
        assert 'pdf' in result, "Should have combined PDF"
        assert 'contour_count' in result, "Should have contour count"
        
        # Check files exist
        for key, file_path in result.items():
            if key != 'contour_count' and file_path:
                assert os.path.exists(file_path), f"File should exist: {file_path}"
                file_size = os.path.getsize(file_path)
                logger.info(f"✅ {key}: {file_size:,} bytes")
        
        # Check contour count
        contour_count = result['contour_count']
        logger.info(f"🎯 Selected {contour_count} contours for outline")
        assert contour_count > 0, "Should have at least one contour"
        
        # Verify SVG content
        if result['svg']:
            with open(result['svg'], 'r') as f:
                svg_content = f.read()
                assert 'image' in svg_content, "SVG should contain raster image layer"
                assert 'cutline-paths' in svg_content, "SVG should contain cutline paths"
                assert 'stroke="#FF00FF"' in svg_content, "SVG should contain pink stroke"
                logger.info("✅ SVG contains both raster and vector layers")
        
        logger.info("🎉 Enhanced contour cutline test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced contour cutline test failed: {str(e)}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)
        
        # Cleanup result files
        if 'result' in locals():
            for key, file_path in result.items():
                if key != 'contour_count' and file_path and os.path.exists(file_path):
                    os.unlink(file_path)

def test_contour_cutline_integration():
    """Test contour cutline integration with main processing"""
    logger = setup_logging()
    logger.info("🧪 Testing contour cutline integration")
    
    # Create test logo
    test_logo = create_test_logo_with_nested_contours()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Initialize processor
        processor = LogoProcessor()
        
        # Test with main processing
        logger.info("🔍 Testing contour cutline with main processing...")
        result = processor.process_logo(test_file, {'contour_cut': True})
        
        # Verify results
        assert result['success'], "Processing should be successful"
        assert 'contour_cut' in result['outputs'], "Should have contour cut output"
        
        contour_output = result['outputs']['contour_cut']
        assert isinstance(contour_output, dict), "Contour output should be a dictionary"
        assert 'contour_count' in contour_output, "Should have contour count"
        
        logger.info(f"🎯 Integration test: {contour_output['contour_count']} contours selected")
        logger.info("🎉 Contour cutline integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Contour cutline integration test failed: {str(e)}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)

def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("🚀 Starting enhanced contour cutline tests")
    
    tests = [
        ("Enhanced Contour Cutline", test_enhanced_contour_cutline),
        ("Contour Cutline Integration", test_contour_cutline_integration)
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
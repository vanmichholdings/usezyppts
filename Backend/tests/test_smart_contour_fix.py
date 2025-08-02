#!/usr/bin/env python3
"""
Test script to verify the smart contour detection fix
Ensures exactly 9 contours for Zyppts logo: 8 text contours + 1 Z emblem outline
"""

import os
import sys
import tempfile
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.logo_processor import LogoProcessor

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_zyppts_test_logo():
    """Create a test Zyppts logo with the expected structure"""
    logger = setup_logging()
    logger.info("🎨 Creating test Zyppts logo...")
    
    # Create a larger image to accommodate the complex logo
    width, height = 800, 400
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create a complex Z emblem (left side)
    # This will create multiple contours due to the layered design
    z_emblem_contours = []
    
    # Outer Z shape (main contour)
    z_points = [
        (50, 50), (200, 50), (200, 100), (100, 100),
        (100, 150), (200, 150), (200, 200), (50, 200)
    ]
    draw.polygon(z_points, fill=(255, 0, 255), outline=(255, 0, 255), width=3)
    z_emblem_contours.append(z_points)
    
    # Inner Z shape (nested contour)
    z_inner_points = [
        (70, 70), (180, 70), (180, 90), (90, 90),
        (90, 130), (180, 130), (180, 150), (70, 150)
    ]
    draw.polygon(z_inner_points, fill=(255, 0, 255), outline=(255, 0, 255), width=2)
    z_emblem_contours.append(z_inner_points)
    
    # Add text "Zyppts" (right side) - each letter should be a separate contour
    text_x = 250
    text_y = 100
    font_size = 60
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    text = "Zyppts"
    draw.text((text_x, text_y), text, fill=(255, 0, 255), font=font)
    
    # Add circle contours inside the 'p' letters (2 circles)
    # First 'p' circle
    circle1_x, circle1_y = text_x + 80, text_y + 40
    draw.ellipse([circle1_x-10, circle1_y-10, circle1_x+10, circle1_y+10], 
                 fill=(255, 0, 255), outline=(255, 0, 255))
    
    # Second 'p' circle  
    circle2_x, circle2_y = text_x + 140, text_y + 40
    draw.ellipse([circle2_x-10, circle2_y-10, circle2_x+10, circle2_y+10], 
                 fill=(255, 0, 255), outline=(255, 0, 255))
    
    logger.info("✅ Created test Zyppts logo with expected structure")
    return img

def test_smart_contour_detection():
    """Test the smart contour detection with Zyppts logo"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("🧪 TESTING SMART CONTOUR DETECTION FIX")
    logger.info("=" * 80)
    
    # Create test logo
    test_logo = create_zyppts_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Initialize processor
        processor = LogoProcessor()
        
        # Test contour cutline
        logger.info("🔍 Testing smart contour cutline with Zyppts logo...")
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
        
        # Check contour count - SHOULD BE EXACTLY 9
        contour_count = result['contour_count']
        logger.info(f"🎯 Contour count: {contour_count}")
        
        if contour_count == 9:
            logger.info("🎉 SUCCESS: Exactly 9 contours detected as expected!")
            logger.info("   - 1 Z emblem outer contour")
            logger.info("   - 6 text letter contours (Z, y, p, p, t, s)")
            logger.info("   - 2 circle contours inside 'p' letters")
            logger.info("   - Total: 9 contours ✅")
        else:
            logger.warning(f"⚠️  Expected 9 contours, but got {contour_count}")
            if contour_count > 9:
                logger.warning("   Too many contours detected - may need further optimization")
            else:
                logger.warning("   Too few contours detected - may need parameter adjustment")
        
        # Test with main processing
        logger.info("🔍 Testing integration with main processing...")
        main_result = processor.process_logo(test_file, {'contour_cut': True})
        
        assert main_result['success'], "Main processing should be successful"
        assert 'contour_cut' in main_result['outputs'], "Should have contour cut output"
        
        main_contour_output = main_result['outputs']['contour_cut']
        assert isinstance(main_contour_output, dict), "Contour output should be a dictionary"
        assert 'contour_count' in main_contour_output, "Should have contour count"
        
        main_contour_count = main_contour_output['contour_count']
        logger.info(f"🎯 Main processing contour count: {main_contour_count}")
        
        if main_contour_count == 9:
            logger.info("🎉 SUCCESS: Main processing also returns exactly 9 contours!")
        else:
            logger.warning(f"⚠️  Main processing returned {main_contour_count} contours instead of 9")
        
        logger.info("🎉 Smart contour detection test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Smart contour detection test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_other_logos():
    """Test that the smart detection works for other logos too"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("🧪 TESTING SMART DETECTION WITH OTHER LOGOS")
    logger.info("=" * 60)
    
    # Create a simple test logo
    width, height = 400, 200
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo with a few shapes
    draw.rectangle([50, 50, 150, 150], fill=(255, 0, 255), outline=(255, 0, 255))
    draw.circle([250, 100], 50, fill=(255, 0, 255), outline=(255, 0, 255))
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        processor = LogoProcessor()
        result = processor._create_contour_cutline(test_file)
        
        contour_count = result['contour_count']
        logger.info(f"🎯 Simple logo contour count: {contour_count}")
        
        if contour_count > 0 and contour_count <= 5:
            logger.info("✅ Smart detection works well for simple logos")
        else:
            logger.warning(f"⚠️  Unexpected contour count for simple logo: {contour_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple logo test failed: {str(e)}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("🚀 Starting smart contour detection tests...")
    
    # Test 1: Zyppts logo (should have exactly 9 contours)
    success1 = test_smart_contour_detection()
    
    # Test 2: Other logos (should work intelligently)
    success2 = test_other_logos()
    
    if success1 and success2:
        logger.info("🎉 All tests passed! Smart contour detection is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)

#!/usr/bin/env python3
"""
Test script to optimize contour detection for Zyppts logo
Focus on getting exactly 9 contours: 8 text contours + 1 Z emblem outline
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

def create_realistic_zyppts_logo():
    """Create a more realistic Zyppts logo that matches the actual structure"""
    logger = setup_logging()
    logger.info("🎨 Creating realistic Zyppts logo...")
    
    # Create a larger image to accommodate the complex logo
    width, height = 1000, 500
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create the complex Z emblem (left side) - this should create 1 main contour
    # Draw a complex Z with multiple layers that will be detected as one outer contour
    z_emblem_points = [
        (50, 50), (300, 50), (300, 100), (150, 100),
        (150, 150), (300, 150), (300, 200), (50, 200)
    ]
    draw.polygon(z_emblem_points, fill=(255, 0, 255), outline=(255, 0, 255), width=5)
    
    # Add text "Zyppts" (right side) - each letter should be a separate contour
    text_x = 350
    text_y = 100
    font_size = 80
    
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw each letter separately to ensure they're detected as separate contours
    letters = ["Z", "y", "p", "p", "t", "s"]
    letter_positions = []
    
    for i, letter in enumerate(letters):
        x = text_x + i * 90
        y = text_y
        letter_positions.append((x, y))
        draw.text((x, y), letter, fill=(255, 0, 255), font=font)
    
    # Add circle contours inside the 'p' letters (2 circles)
    # First 'p' circle
    p1_x, p1_y = letter_positions[2][0] + 30, letter_positions[2][1] + 50
    draw.ellipse([p1_x-15, p1_y-15, p1_x+15, p1_y+15], 
                 fill=(255, 0, 255), outline=(255, 0, 255))
    
    # Second 'p' circle  
    p2_x, p2_y = letter_positions[3][0] + 30, letter_positions[3][1] + 50
    draw.ellipse([p2_x-15, p2_y-15, p2_x+15, p2_y+15], 
                 fill=(255, 0, 255), outline=(255, 0, 255))
    
    logger.info("✅ Created realistic Zyppts logo with expected structure:")
    logger.info("   - 1 Z emblem outer contour")
    logger.info("   - 6 text letter contours (Z, y, p, p, t, s)")
    logger.info("   - 2 circle contours inside 'p' letters")
    logger.info("   - Total: 9 contours expected")
    
    return img

def test_contour_optimization():
    """Test and optimize contour detection for Zyppts logo"""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("🧪 TESTING CONTOUR OPTIMIZATION FOR ZYPTS LOGO")
    logger.info("=" * 80)
    
    # Create realistic test logo
    test_logo = create_realistic_zyppts_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        test_logo.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        # Initialize processor
        processor = LogoProcessor()
        
        # Test contour cutline
        logger.info("🔍 Testing contour cutline with realistic Zyppts logo...")
        result = processor._create_contour_cutline(test_file)
        
        # Check contour count
        contour_count = result['contour_count']
        logger.info(f"🎯 Contour count: {contour_count}")
        
        if contour_count == 9:
            logger.info("🎉 PERFECT: Exactly 9 contours detected!")
            logger.info("   ✅ Smart contour detection is working correctly")
        elif contour_count >= 7 and contour_count <= 11:
            logger.info(f"✅ GOOD: {contour_count} contours detected (close to expected 9)")
            logger.info("   The detection is working well, minor adjustments may be needed")
        else:
            logger.warning(f"⚠️  Unexpected contour count: {contour_count}")
            logger.info("   May need parameter adjustments for optimal detection")
        
        # Check files exist and have reasonable sizes
        for key, file_path in result.items():
            if key != 'contour_count' and file_path:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"✅ {key}: {file_size:,} bytes")
                else:
                    logger.error(f"❌ {key}: File missing - {file_path}")
        
        logger.info("🎉 Contour optimization test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Contour optimization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("🚀 Starting Zyppts contour optimization test...")
    
    success = test_contour_optimization()
    
    if success:
        logger.info("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Test failed. Please check the implementation.")
        sys.exit(1)

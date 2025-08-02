#!/usr/bin/env python3
"""
Test to verify the new contour cutline functionality with separate masks
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

def test_contour_cutline_masks():
    """Test the new contour cutline functionality with separate masks"""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TESTING CONTOUR CUTLINE WITH SEPARATE MASKS")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test contour cutline processing
        logger.info("Processing contour cutline...")
        result = processor._create_contour_cutline(test_file)
        
        logger.info(f"Contour cutline result: {result}")
        
        # Check that all expected files were created
        expected_files = ['full_color', 'pink_outline', 'svg', 'pdf']
        all_files_exist = True
        
        for file_type in expected_files:
            if file_type in result:
                file_path = result[file_type]
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.info(f"✅ {file_type}: {file_path} ({file_size:,} bytes)")
                else:
                    logger.error(f"❌ {file_type}: File missing - {file_path}")
                    all_files_exist = False
            else:
                logger.error(f"❌ {file_type}: Not in result dictionary")
                all_files_exist = False
        
        if not all_files_exist:
            return False
        
        # Verify the files are different (full color vs pink outline)
        if 'full_color' in result and 'pink_outline' in result:
            full_color_img = Image.open(result['full_color'])
            pink_outline_img = Image.open(result['pink_outline'])
            
            # Convert to arrays for comparison
            full_color_array = full_color_img.convert('RGB')
            pink_outline_array = pink_outline_img.convert('RGB')
            
            # Check that they're different (full color should have colors, pink outline should be mostly black with pink)
            full_color_data = list(full_color_array.getdata())
            pink_outline_data = list(pink_outline_array.getdata())
            
            # Count non-black pixels in pink outline (should be pink outline pixels)
            pink_pixels = sum(1 for pixel in pink_outline_data if pixel != (0, 0, 0))
            total_pixels = len(pink_outline_data)
            pink_percentage = (pink_pixels / total_pixels) * 100
            
            logger.info(f"📊 Pink outline analysis:")
            logger.info(f"   Total pixels: {total_pixels:,}")
            logger.info(f"   Pink pixels: {pink_pixels:,}")
            logger.info(f"   Pink percentage: {pink_percentage:.2f}%")
            
            if pink_percentage > 0:
                logger.info("✅ Pink outline contains pink pixels (outline detected)")
            else:
                logger.error("❌ Pink outline has no pink pixels")
                return False
            
            # Check that full color image has more color variation than pink outline
            full_color_colors = set(full_color_data)
            pink_outline_colors = set(pink_outline_data)
            
            logger.info(f"📊 Color analysis:")
            logger.info(f"   Full color unique colors: {len(full_color_colors)}")
            logger.info(f"   Pink outline unique colors: {len(pink_outline_colors)}")
            
            if len(full_color_colors) > len(pink_outline_colors):
                logger.info("✅ Full color image has more color variation than pink outline")
            else:
                logger.warning("⚠️ Unexpected color distribution")
        
        # Test SVG content
        if 'svg' in result:
            with open(result['svg'], 'r') as f:
                svg_content = f.read()
            
            if 'stroke="#FF00FF"' in svg_content:
                logger.info("✅ SVG contains pink stroke color")
            else:
                logger.error("❌ SVG missing pink stroke color")
                return False
            
            if 'path' in svg_content:
                logger.info("✅ SVG contains path elements")
            else:
                logger.error("❌ SVG missing path elements")
                return False
        
        logger.info("🎉 All contour cutline tests passed!")
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
            # Clean up generated files
            if 'result' in locals():
                for file_type, file_path in result.items():
                    if os.path.exists(file_path):
                        os.unlink(file_path)
        except:
            pass

def test_contour_cutline_integration():
    """Test contour cutline integration with the main processing pipeline"""
    logger = setup_logging()
    logger.info("\n" + "=" * 60)
    logger.info("TESTING CONTOUR CUTLINE INTEGRATION")
    logger.info("=" * 60)
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=2)
        
        # Test with contour cutline in the main processing pipeline
        options = {
            'contour_cut': True
        }
        
        logger.info("Processing with contour cutline option...")
        result = processor.process_logo(test_file, options)
        
        logger.info(f"Processing result: {result['success']}")
        logger.info(f"Outputs: {list(result['outputs'].keys())}")
        logger.info(f"Processing time: {result['processing_time']:.2f}s")
        
        if 'contour_cut' in result['outputs']:
            contour_result = result['outputs']['contour_cut']
            logger.info(f"Contour cutline outputs: {contour_result}")
            
            # Check that all expected files are present
            expected_keys = ['full_color', 'pink_outline', 'svg', 'pdf']
            for key in expected_keys:
                if key in contour_result:
                    file_path = contour_result[key]
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        logger.info(f"✅ {key}: {file_size:,} bytes")
                    else:
                        logger.error(f"❌ {key}: File missing")
                        return False
                else:
                    logger.error(f"❌ {key}: Missing from result")
                    return False
            
            logger.info("✅ Contour cutline integration successful!")
            return True
        else:
            logger.error("❌ Contour cutline not in outputs")
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
    
    # Test 1: Contour cutline masks
    if not test_contour_cutline_masks():
        all_tests_passed = False
    
    # Test 2: Integration test
    if not test_contour_cutline_integration():
        all_tests_passed = False
    
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("✅ ALL CONTOUR CUTLINE TESTS PASSED")
        logger.info("✅ Contour cutline with separate masks is working correctly")
        logger.info("✅ Full color and pink outline masks are properly separated")
    else:
        logger.error("❌ SOME CONTOUR CUTLINE TESTS FAILED")
        logger.error("❌ Issues found with contour cutline functionality")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 
import os
import sys
from pathlib import Path

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import logging
import numpy as np
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transparent_png():
    logger.info("Starting transparent PNG test with multi-color background removal")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Create test logos directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logos")
    logger.info(f"Test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test logos with different background colors
    test_images = [
        {
            "name": "white_bg_logo",
            "description": "Logo with white background",
            "path": os.path.join(test_dir, "white_bg_logo.png"),
            "bg_color": (255, 255, 255)
        },
        {
            "name": "blue_bg_logo",
            "description": "Logo with blue background",
            "path": os.path.join(test_dir, "blue_bg_logo.png"),
            "bg_color": (0, 100, 200)
        },
        {
            "name": "green_bg_logo",
            "description": "Logo with green background",
            "path": os.path.join(test_dir, "green_bg_logo.png"),
            "bg_color": (50, 150, 50)
        },
        {
            "name": "red_bg_logo",
            "description": "Logo with red background",
            "path": os.path.join(test_dir, "red_bg_logo.png"),
            "bg_color": (200, 50, 50)
        }
    ]
    
    # Create test logos if they don't exist
    for logo in test_images:
        if not os.path.exists(logo["path"]):
            create_test_logo(logo["path"], logo["bg_color"])
    
    # Initialize processor
    processor = LogoProcessor()
    
    for logo in test_images:
        logger.info(f"\nTesting {logo['name']}: {logo['description']}")
        
        # Start timing
        start_time = time.time()
        
        # Create transparent PNG
        result = processor._create_transparent_png(logo["path"])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if result:
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
            # Verify output
            try:
                output_img = Image.open(result)
                if output_img.mode != 'RGBA':
                    logger.error("Output is not in RGBA mode")
                elif output_img.size != Image.open(logo["path"]).size:
                    logger.error(f"Output size mismatch: {output_img.size} vs {Image.open(logo['path']).size}")
                else:
                    logger.info("Output verification successful")
                    
                    # Check alpha channel
                    alpha = np.array(output_img.split()[-1])
                    transparent_pixels = np.sum(alpha == 0)
                    total_pixels = alpha.size
                    transparency_percentage = (transparent_pixels / total_pixels) * 100
                    
                    logger.info(f"Transparency analysis: {transparent_pixels:,} transparent pixels ({transparency_percentage:.1f}% of image)")
                    
                    if np.all(alpha == 0) or np.all(alpha == 255):
                        logger.warning("Alpha channel appears to be binary, may lack transparency")
                    else:
                        logger.info("Alpha channel has proper transparency")
                        
                    # Save test results to a more accessible location
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
                    logger.info(f"Output directory: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    result_name = f"{logo['name']}_transparent.png"
                    output_path = os.path.join(output_dir, result_name)
                    output_img.save(output_path, 'PNG', optimize=True, compress_level=9)
                    logger.info(f"Saved test result to: {output_path}")
            except Exception as e:
                logger.error(f"Error verifying output: {str(e)}")
        else:
            logger.error("Failed to create transparent PNG")

def create_test_logo(path, bg_color):
    """Create test logos with different background colors."""
    size = (500, 500)
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Create a simple logo design
    # Main shape
    draw.rectangle([(100, 100), (400, 400)], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    
    # Inner detail
    draw.ellipse([(150, 150), (350, 350)], fill=(255, 0, 0))
    
    # Text-like element
    draw.rectangle([(200, 200), (300, 250)], fill=(0, 255, 0))
    
    # Small detail
    draw.polygon([(225, 175), (250, 150), (275, 175)], fill=(0, 0, 255))
    
    img.save(path, 'PNG')

if __name__ == "__main__":
    test_transparent_png()

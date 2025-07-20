#!/usr/bin/env python

import os
import sys
import logging
from PIL import Image
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the LogoProcessor
from zyppts.utils.logo_processor import LogoProcessor

def test_all_variations():
    """Test all logo variations using the LogoProcessor."""
    logger = logging.getLogger("test_logo_processor")
    logger.info("Starting all-variations logo processor test")

    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp(prefix="logo_test_")
    cache_dir = os.path.join(temp_dir, "cache")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the LogoProcessor
    processor = LogoProcessor(
        cache_dir=cache_dir,
        output_folder=output_dir
    )

    # Path to test image - use the Z logo if available, otherwise use a sample image
    test_image_path = None
    potential_paths = [
        "./zyppt_logo.png",
        "./zyppts_logo.png",
        "./logo.png",
        "./test_images/logo.png"
    ]
    for path in potential_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    if test_image_path is None:
        logger.warning("No test image found. Please provide a path to a test image.")
        test_image_path = input("Enter path to test image: ")

    # Set options to generate all variations
    options = {
        'transparent_png': True,
        'black_version': True,
        'distressed_effect': True,
        'vector_trace': True,
        'contour_cut': True,
        'color_separations': True,
        'logo_upscaler': True,
        # Add more options if needed
    }

    # Process the image to generate all variations
    logger.info(f"Processing image {test_image_path} to generate all variations...")
    result = processor.process_logo(file_path=test_image_path, options=options)

    if result.get('success'):
        logger.info(f"Successfully generated variations: {result.get('message')}")
        outputs = result.get('outputs', {})
        for key, value in outputs.items():
            logger.info(f"Variation: {key} -> {value}")
            print(f"Variation: {key} -> {value}")
    else:
        logger.error(f"Failed to generate variations: {result.get('message')}")
        errors = result.get('errors', [])
        for err in errors:
            logger.error(f"Error: {err}")
            print(f"Error: {err}")

    logger.info(f"All outputs saved to: {os.path.abspath(output_dir)}")
    print(f"All outputs saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    test_all_variations()

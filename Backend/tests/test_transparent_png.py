import os
import time
import logging
from PIL import Image, ImageDraw
from zyppts.utils.logo_processor import LogoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transparent_png():
    logger.info("Starting transparent PNG test")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Create test logos directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logos")
    logger.info(f"Test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test logos with different characteristics
    test_images = [
        {
            "name": "solid_logo",
            "description": "Solid color logo with simple shape",
            "path": os.path.join(test_dir, "solid_logo.png")
        },
        {
            "name": "gradient_logo",
            "description": "Logo with gradient and soft edges",
            "path": os.path.join(test_dir, "gradient_logo.png")
        },
        {
            "name": "complex_logo",
            "description": "Logo with multiple colors and fine details",
            "path": os.path.join(test_dir, "complex_logo.png")
        }
    ]
    
    # Create test logos if they don't exist
    for logo in test_images:
        if not os.path.exists(logo["path"]):
            create_test_logo(logo["path"])
    
    # Initialize processor
    processor = LogoProcessor()
    
    # Test different sensitivity levels
    sensitivities = [0.2, 0.5, 0.8]
    
    for logo in test_images:
        logger.info(f"\nTesting {logo['name']}: {logo['description']}")
        
        # Load test image
        img = Image.open(logo["path"])
        
        for sensitivity in sensitivities:
            logger.info(f"\nTesting sensitivity: {sensitivity}")
            
            # Start timing
            start_time = time.time()
            
            # Create transparent PNG
            result = processor._create_transparent_png(img, sensitivity=sensitivity)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            if result:
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                
                # Verify output
                try:
                    output_img = Image.open(result)
                    if output_img.mode != 'RGBA':
                        logger.error("Output is not in RGBA mode")
                    elif output_img.size != img.size:
                        logger.error(f"Output size mismatch: {output_img.size} vs {img.size}")
                    else:
                        logger.info("Output verification successful")
                        
                        # Check alpha channel
                        alpha = np.array(output_img.split()[-1])
                        if np.all(alpha == 0) or np.all(alpha == 255):
                            logger.warning("Alpha channel appears to be binary, may lack transparency")
                        else:
                            logger.info("Alpha channel has proper transparency")
                            
                        # Save test results to a more accessible location
                        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
                        logger.info(f"Output directory: {output_dir}")
                        os.makedirs(output_dir, exist_ok=True)
                        result_name = f"{logo['name']}_sensitivity_{sensitivity}.png"
                        output_path = os.path.join(output_dir, result_name)
                        output_img.save(output_path, 'PNG', optimize=True, compress_level=9)
                        logger.info(f"Saved test result to: {output_path}")
                except Exception as e:
                    logger.error(f"Error verifying output: {str(e)}")
            else:
                logger.error("Failed to create transparent PNG")

def create_test_logo(path):
    """Create test logos with different characteristics."""
    size = (500, 500)
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    if "solid" in path:
        # Create solid color logo
        draw.rectangle([(100, 100), (400, 400)], fill=(0, 0, 255))
    elif "gradient" in path:
        # Create gradient logo
        for y in range(size[1]):
            color = (y % 255, 0, 255 - (y % 255))
            draw.line([(0, y), (size[0], y)], fill=color)
    else:
        # Create complex logo
        draw.rectangle([(100, 100), (400, 400)], fill=(255, 0, 0))
        draw.ellipse([(150, 150), (350, 350)], fill=(0, 255, 0))
        draw.polygon([(200, 200), (250, 150), (300, 200)], fill=(0, 0, 255))
    
    img.save(path, 'PNG')

if __name__ == "__main__":
    test_transparent_png()

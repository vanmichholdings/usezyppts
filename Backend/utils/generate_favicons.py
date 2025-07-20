import os
from PIL import Image
import numpy as np

def generate_favicons(source_image_path, output_dir):
    """Generate favicon files in different sizes."""
    # Favicon sizes needed
    sizes = [
        (16, 16),
        (32, 32),
        (48, 48),
        (57, 57),
        (60, 60),
        (72, 72),
        (76, 76),
        (96, 96),
        (114, 114),
        (120, 120),
        (144, 144),
        (152, 152),
        (180, 180),
        (192, 192),
        (512, 512),
    ]
    
    # Open the source image
    img = Image.open(source_image_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each size
    for size in sizes:
        resized = img.copy()
        resized.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create output filename
        if size[0] == size[1]:
            filename = f"favicon-{size[0]}x{size[1]}.png"
        else:
            filename = f"apple-touch-icon-{size[0]}x{size[1]}.png"
        
        # Save the resized image
        output_path = os.path.join(output_dir, filename)
        resized.save(output_path, "PNG", optimize=True)
        
        # Special cases
        if size == (180, 180):
            # Save as apple-touch-icon.png
            resized.save(os.path.join(output_dir, "apple-touch-icon.png"), "PNG", optimize=True)
        elif size == (32, 32):
            # Save as favicon.ico
            resized.save(os.path.join(output_dir, "favicon.ico"), "ICO")

if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    source_image = os.path.join(project_root, "static", "images", "logo", "zyppts-emblem.png")
    output_dir = os.path.join(project_root, "static", "images", "favicon")
    
    # Generate favicons
    generate_favicons(source_image, output_dir) 
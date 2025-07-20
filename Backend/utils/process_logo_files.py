from PIL import Image
import numpy as np
import os

def process_logo_files(input_path, output_dir):
    """Process the logo file and create the emblem."""
    try:
        # Create output directories
        logo_dir = os.path.join(output_dir, 'logo')
        os.makedirs(logo_dir, exist_ok=True)
        
        # Open the original logo
        img = Image.open(input_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Extract the emblem (Z part)
        data = np.array(img)
        alpha = data[:, :, 3]
        
        # Find the bounding box of visible content
        rows, cols = np.nonzero(alpha)
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("No visible content found in image")
        
        # Calculate dimensions
        width = cols.max() - cols.min()
        emblem_width = width // 3  # Take roughly the first third which contains the Z
        
        # Calculate the bounding box
        left = cols.min()
        right = left + emblem_width
        top = rows.min()
        bottom = rows.max()
        
        # Extract and save the emblem
        emblem = img.crop((left, top, right, bottom))
        emblem_path = os.path.join(logo_dir, 'zyppts-emblem.png')
        emblem.save(emblem_path, 'PNG', optimize=True)
        print(f"Saved emblem to: {emblem_path}")
        
        # Copy original logo to logo directory
        logo_path = os.path.join(logo_dir, 'zyppts-logo.png')
        img.save(logo_path, 'PNG', optimize=True)
        print(f"Saved logo to: {logo_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing logo files: {str(e)}")
        return False

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python process_logo_files.py <path_to_original_logo>")
        sys.exit(1)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_path = sys.argv[1]
    output_dir = os.path.join(project_root, 'static', 'images')
    
    # Process the logo
    success = process_logo_files(input_path, output_dir)
    sys.exit(0 if success else 1) 
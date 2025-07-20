import os
import sys
from process_logo_files import process_logo_files
from generate_favicons import generate_favicons

def setup_logo_assets(logo_path):
    """Process logo and generate all necessary assets."""
    try:
        # Get project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        static_dir = os.path.join(project_root, 'static', 'images')
        
        # Process logo files
        print("Processing logo files...")
        success = process_logo_files(logo_path, static_dir)
        if not success:
            print("Failed to process logo files")
            return False
        
        # Generate favicons
        print("\nGenerating favicons...")
        emblem_path = os.path.join(static_dir, 'logo', 'zyppts-emblem.png')
        favicon_dir = os.path.join(static_dir, 'favicon')
        
        if not os.path.exists(emblem_path):
            print(f"Emblem file not found at {emblem_path}")
            return False
            
        generate_favicons(emblem_path, favicon_dir)
        print("Favicon generation complete!")
        
        return True
        
    except Exception as e:
        print(f"Error setting up logo assets: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python setup_logo_assets.py <path_to_logo_file>")
        sys.exit(1)
        
    logo_path = sys.argv[1]
    if not os.path.exists(logo_path):
        print(f"Logo file not found: {logo_path}")
        sys.exit(1)
        
    success = setup_logo_assets(logo_path)
    sys.exit(0 if success else 1) 
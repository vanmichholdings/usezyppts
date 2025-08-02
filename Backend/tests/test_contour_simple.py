#!/usr/bin/env python3
"""Simple test for contour cutline functionality"""

import tempfile
import os
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def main():
    # Create test logo
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.rectangle([60, 60, 140, 140], fill=(0, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img.save(f.name, 'PNG')
        test_file = f.name
    
    try:
        processor = LogoProcessor()
        result = processor.process_logo(test_file, {'contour_cut': True})
        
        print('Success:', result['success'])
        print('Outputs:', list(result['outputs'].keys()))
        
        if 'contour_cut' in result['outputs']:
            contour_files = result['outputs']['contour_cut']
            print('Contour cut files:')
            for file_type, file_path in contour_files.items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f'  {file_type}: {file_size:,} bytes')
                else:
                    print(f'  {file_type}: MISSING')
        else:
            print('Contour cut files: None')
            
    finally:
        os.unlink(test_file)

if __name__ == "__main__":
    main() 
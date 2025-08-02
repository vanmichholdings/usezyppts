#!/usr/bin/env python3
"""
Convert Logo to Base64 for Email Templates
"""

import base64
import os
from pathlib import Path

def convert_logo_to_base64():
    """Convert logo image to base64 string"""
    print("🖼️ Converting Logo to Base64")
    print("=" * 60)
    
    # Path to the logo file
    logo_path = Path(__file__).parent.parent.parent / 'Frontend' / 'static' / 'images' / 'logo' / 'zyppts-logo-new.png'
    
    if not logo_path.exists():
        print(f"❌ Logo file not found: {logo_path}")
        return None
    
    try:
        # Read the image file
        with open(logo_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Convert to base64
        base64_string = base64.b64encode(image_data).decode('utf-8')
        
        # Create the data URL
        data_url = f"data:image/png;base64,{base64_string}"
        
        print(f"✅ Logo converted successfully")
        print(f"📏 Original size: {len(image_data)} bytes")
        print(f"📏 Base64 size: {len(base64_string)} characters")
        
        # Save to a file for easy copying
        output_file = Path(__file__).parent / 'logo_base64.txt'
        with open(output_file, 'w') as f:
            f.write(data_url)
        
        print(f"💾 Base64 data saved to: {output_file}")
        
        return data_url
        
    except Exception as e:
        print(f"❌ Error converting logo: {e}")
        return None

def get_logo_base64():
    """Get the base64 logo data"""
    # Try to read from existing file first
    base64_file = Path(__file__).parent / 'logo_base64.txt'
    
    if base64_file.exists():
        with open(base64_file, 'r') as f:
            return f.read().strip()
    
    # If file doesn't exist, convert the logo
    return convert_logo_to_base64()

if __name__ == "__main__":
    convert_logo_to_base64() 
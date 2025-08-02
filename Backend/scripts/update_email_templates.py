#!/usr/bin/env python3
"""
Update Email Templates with Base64 Logo and Brand Colors
"""

import os
import re
from pathlib import Path

def get_base64_logo():
    """Get the base64 logo data"""
    base64_file = Path(__file__).parent / 'logo_base64.txt'
    
    if base64_file.exists():
        with open(base64_file, 'r') as f:
            return f.read().strip()
    else:
        print("❌ Base64 logo file not found. Run convert_logo_to_base64.py first.")
        return None

def update_email_template(template_path, base64_logo):
    """Update a single email template"""
    print(f"📝 Updating {template_path.name}...")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the logo placeholder with base64 data
    content = content.replace(
        'src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."',
        f'src="{base64_logo}"'
    )
    
    # Replace "Zyppts" text with brand orange styling
    content = re.sub(
        r'Zyppts(?!.*class="brand-text")',
        '<span class="brand-text">Zyppts</span>',
        content
    )
    
    # Add brand-text CSS class if not present
    if '.brand-text' not in content:
        # Find the closing </style> tag and add the CSS before it
        style_end = content.find('</style>')
        if style_end != -1:
            brand_css = '''
        .brand-text {
            color: #FFB81C;
            font-weight: 600;
        }'''
            content = content[:style_end] + brand_css + content[style_end:]
    
    # Write the updated content back
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {template_path.name}")

def main():
    """Update all email templates"""
    print("🎨 Updating Email Templates with Brand Colors")
    print("=" * 60)
    
    # Get the base64 logo data
    base64_logo = get_base64_logo()
    if not base64_logo:
        return
    
    # Find all email templates
    templates_dir = Path(__file__).parent.parent.parent / 'Frontend' / 'templates' / 'emails'
    email_templates = list(templates_dir.glob('*.html'))
    
    print(f"📁 Found {len(email_templates)} email templates")
    
    # Update each template
    for template in email_templates:
        update_email_template(template, base64_logo)
    
    print("\n✅ All email templates updated successfully!")
    print("\n📋 Changes made:")
    print("- Added base64 encoded logo (no remote content)")
    print("- Changed 'Zyppts' text to brand orange (#FFB81C)")
    print("- Added brand-text CSS class for consistent styling")

if __name__ == "__main__":
    main() 
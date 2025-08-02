#!/usr/bin/env python3
"""
Fix Email Templates with Base64 Logo and Brand Colors
"""

import os
import re
from pathlib import Path

def fix_email_templates():
    """Fix all email templates"""
    print("🎨 Fixing Email Templates")
    print("=" * 60)
    
    # Get base64 logo
    base64_file = Path(__file__).parent / 'logo_base64.txt'
    if not base64_file.exists():
        print("❌ Base64 logo file not found")
        return
    
    with open(base64_file, 'r') as f:
        base64_logo = f.read().strip()
    
    # Templates to update
    templates = [
        'Frontend/templates/emails/welcome_email.html',
        'Frontend/templates/emails/new_account_notification.html',
        'Frontend/templates/emails/test_notification.html',
        'Frontend/templates/emails/daily_summary.html',
        'Frontend/templates/emails/weekly_report.html'
    ]
    
    for template_path in templates:
        full_path = Path(__file__).parent.parent / template_path
        if full_path.exists():
            print(f"📝 Updating {template_path}...")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace logo placeholder with base64
            content = content.replace(
                'src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."',
                f'src="{base64_logo}"'
            )
            
            # Replace text logo with image logo
            content = content.replace(
                '<div class="logo-text">Zyppts</div>',
                f'<img src="{base64_logo}" alt="Zyppts Logo" class="logo">'
            )
            
            # Add brand-text CSS if not present
            if '.brand-text' not in content:
                # Find </style> and add CSS before it
                style_end = content.find('</style>')
                if style_end != -1:
                    brand_css = '''
        .brand-text {
            color: #FFB81C;
            font-weight: 600;
        }'''
                    content = content[:style_end] + brand_css + content[style_end:]
            
            # Replace "Zyppts" with brand orange styling
            content = re.sub(
                r'Zyppts(?!.*class="brand-text")',
                '<span class="brand-text">Zyppts</span>',
                content
            )
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated {template_path}")
        else:
            print(f"⚠️ Template not found: {template_path}")
    
    print("\n✅ All email templates updated!")
    print("📋 Changes made:")
    print("- Added base64 encoded logo (no remote content)")
    print("- Changed 'Zyppts' text to brand orange (#FFB81C)")
    print("- Added brand-text CSS class")

if __name__ == "__main__":
    fix_email_templates() 
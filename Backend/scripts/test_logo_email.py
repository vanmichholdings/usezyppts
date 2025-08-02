#!/usr/bin/env python3
"""
Test Email with Base64 Logo
"""

import os
import sys
from pathlib import Path

def test_logo_email():
    """Test email with base64 logo"""
    print("🧪 Testing Email with Base64 Logo")
    print("=" * 60)
    
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.append(str(backend_dir))
    
    try:
        from app_config import create_app
        from utils.email_notifications import send_email
        from datetime import datetime
        
        app = create_app()
        
        with app.app_context():
            # Get base64 logo
            base64_file = Path(__file__).parent / 'logo_base64.txt'
            if base64_file.exists():
                with open(base64_file, 'r') as f:
                    base64_logo = f.read().strip()
                
                # Create a simple HTML email with the logo
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .brand-text {{ color: #FFB81C; font-weight: 600; }}
                        .logo {{ max-width: 200px; height: auto; }}
                    </style>
                </head>
                <body>
                    <div style="text-align: center; padding: 20px;">
                        <img src="{base64_logo}" alt="Zyppts Logo" class="logo">
                        <h1>Welcome to <span class="brand-text">Zyppts</span>!</h1>
                        <p>This is a test email with the embedded logo and brand orange text.</p>
                        <p>The <span class="brand-text">Zyppts</span> logo should display properly without being blocked as remote content.</p>
                    </div>
                </body>
                </html>
                """
                
                # Send test email
                print("📤 Sending test email with base64 logo...")
                
                # Use Flask-Mail directly to send custom HTML
                from flask_mail import Message
                from app_config import mail
                
                msg = Message(
                    subject="🧪 Logo Test - Zyppts",
                    recipients=[app.config.get('ADMIN_ALERT_EMAIL', 'zyppts@gmail.com')],
                    sender=app.config.get('MAIL_DEFAULT_SENDER', 'Zyppts HQ <zyppts@gmail.com>')
                )
                msg.html = html_content
                
                mail.send(msg)
                
                print("✅ Test email sent successfully")
                print("📋 Check your email inbox for the logo test")
                print("  - Logo should display as an image (not blocked)")
                print("  - 'Zyppts' text should be in brand orange (#FFB81C)")
                
            else:
                print("❌ Base64 logo file not found")
                
    except Exception as e:
        print(f"❌ Error testing logo email: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_logo_email() 
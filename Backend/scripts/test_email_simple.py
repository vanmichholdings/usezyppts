#!/usr/bin/env python3
"""
Simple Email Test Script for zyppts@gmail.com
"""

import os
import sys
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def test_email_connection():
    """Test basic email connection to Gmail"""
    print("📧 Testing Email Connection to Gmail")
    print("=" * 50)
    
    # Read password from .env file
    env_file = Path('.env')
    password = "your-app-password-here"  # Default placeholder
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('MAIL_PASSWORD='):
                    password = line.split('=', 1)[1].strip()
                    break
    
    # Email settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    username = "zyppts@gmail.com"
    
    try:
        # Create server connection
        print("🔌 Connecting to Gmail SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        print("🔐 Attempting to authenticate...")
        server.login(username, password)
        print("✅ Authentication successful!")
        
        # Create test email
        msg = MIMEMultipart()
        msg['From'] = "noreply@zyppts.com"
        msg['To'] = "zyppts@gmail.com"
        msg['Subject'] = "🧪 ZYPPTS Email Test - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Email body
        body = f"""
        🎉 ZYPPTS Email System Test
        
        This is a test email to verify that your email configuration is working correctly.
        
        Test Details:
        - Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
        - SMTP Server: {smtp_server}:{smtp_port}
        - From: noreply@zyppts.com
        - To: zyppts@gmail.com
        
        ✅ If you received this email, your email system is working!
        
        Next Steps:
        1. You'll receive notifications when new users register
        2. Daily and weekly reports will be sent automatically
        3. Security alerts will be sent for suspicious activity
        
        Best regards,
        ZYPPTS Admin System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        print("📤 Sending test email...")
        text = msg.as_string()
        server.sendmail("noreply@zyppts.com", "zyppts@gmail.com", text)
        
        print("✅ Test email sent successfully!")
        print("📧 Check zyppts@gmail.com for the test email")
        
        server.quit()
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("❌ Authentication failed!")
        print("⚠️  You need to:")
        print("1. Enable 2-factor authentication on your Google account")
        print("2. Generate an App Password for 'Mail'")
        print("3. Replace 'your-app-password-here' in your .env file with your actual app password")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_gmail_app_password_guide():
    """Create a guide for setting up Gmail App Password"""
    print("\n📖 Gmail App Password Setup Guide")
    print("=" * 50)
    print("""
    To set up Gmail App Password for zyppts@gmail.com:
    
    1. Go to your Google Account settings:
       https://myaccount.google.com/
    
    2. Click on "Security" in the left sidebar
    
    3. Under "Signing in to Google", click on "2-Step Verification"
       - If not enabled, enable it first
    
    4. Scroll down and click on "App passwords"
    
    5. Select "Mail" from the dropdown and click "Generate"
    
    6. Copy the 16-character password that appears
    
    7. Replace 'your-app-password-here' in your .env file with this password
    
    8. Restart your Flask application
    
    The app password will look something like: "abcd efgh ijkl mnop"
    """)

def main():
    """Main function"""
    print("🚀 ZYPPTS Email Configuration Test")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Run the configure_email.py script first.")
        return
    
    print("✅ .env file found")
    
    # Test email connection
    if test_email_connection():
        print("\n🎉 Email system is ready!")
        print("You'll now receive:")
        print("- New user registration notifications")
        print("- Daily summary reports")
        print("- Weekly analytics reports")
        print("- Security alerts")
    else:
        print("\n⚠️  Email system needs configuration")
        create_gmail_app_password_guide()

if __name__ == "__main__":
    main() 
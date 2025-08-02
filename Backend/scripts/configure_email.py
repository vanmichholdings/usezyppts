#!/usr/bin/env python3
"""
Configure Email Settings and Send Test Notifications
"""

import os
import sys
from pathlib import Path

def configure_email_settings():
    """Configure email settings for zyppts@gmail.com"""
    print("📧 Configuring Email Settings for zyppts@gmail.com")
    print("=" * 60)
    
    # Environment variables to set
    env_vars = {
        'MAIL_SERVER': 'smtp.gmail.com',
        'MAIL_PORT': '587',
        'MAIL_USE_TLS': 'True',
        'MAIL_USERNAME': 'zyppts@gmail.com',
        'MAIL_PASSWORD': 'your-app-password-here',  # User needs to set this
        'MAIL_DEFAULT_SENDER': 'Zyppts HQ <zyppts@gmail.com>',
        'ADMIN_ALERT_EMAIL': 'zyppts@gmail.com',
        'SITE_URL': 'http://localhost:5000'
    }
    
    # Create .env file content
    env_content = """# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=zyppts@gmail.com
MAIL_PASSWORD=your-app-password-here
MAIL_DEFAULT_SENDER=Zyppts HQ <zyppts@gmail.com>

# Admin Alert Email
ADMIN_ALERT_EMAIL=zyppts@gmail.com

# Admin Security
ADMIN_IP_WHITELIST=127.0.0.1
ADMIN_ALLOWED_EMAILS=test@zyppts.com
ADMIN_SECRET_KEY=your-super-secret-admin-key-change-this
ADMIN_SESSION_TIMEOUT=86400
ADMIN_RATE_LIMIT=100
ADMIN_MAX_LOGIN_ATTEMPTS=5
ADMIN_LOCKOUT_DURATION=900

# Flask Security
SECRET_KEY=your-flask-secret-key-here
SESSION_COOKIE_SECURE=False
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# Database
DATABASE_URL=sqlite:///app.db

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Site URL
SITE_URL=http://localhost:5000
"""
    
    # Write .env file
    env_file = Path('.env')
    if env_file.exists():
        backup_file = Path('.env.backup')
        env_file.rename(backup_file)
        print(f"💾 Backed up existing .env to .env.backup")
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with email configuration")
    print("\n⚠️  IMPORTANT: You need to set up Gmail App Password")
    print("1. Go to your Google Account settings")
    print("2. Enable 2-factor authentication")
    print("3. Generate an App Password for 'Mail'")
    print("4. Replace 'your-app-password-here' in .env with your actual app password")
    
    return True

def send_test_emails():
    """Send test emails to zyppts@gmail.com"""
    print("\n📤 Sending Test Emails to zyppts@gmail.com")
    print("=" * 60)
    
    # Add the backend directory to the path
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from app_config import create_app
        from utils.email_notifications import send_email, send_new_account_notification
        from models import User
        from datetime import datetime
        
        app = create_app()
        
        with app.app_context():
            # Test 1: Basic email
            print("1️⃣ Sending basic test email...")
            success1 = send_email(
                subject="🧪 Test Email - ZYPPTS Admin",
                recipients=['zyppts@gmail.com'],
                template='test_notification',
                admin_name='Admin Test',
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            )
            
            if success1:
                print("✅ Basic test email sent successfully")
            else:
                print("❌ Failed to send basic test email")
            
            # Test 2: New account notification
            print("2️⃣ Sending new account notification...")
            
            # Create a test user object
            test_user = User(
                username='testuser',
                email='test@example.com',
                created_at=datetime.utcnow(),
                registration_ip='127.0.0.1',
                registration_user_agent='Test Browser'
            )
            
            success2 = send_new_account_notification(test_user)
            
            if success2:
                print("✅ New account notification sent successfully")
            else:
                print("❌ Failed to send new account notification")
            
            # Test 3: Welcome email
            print("3️⃣ Sending welcome email...")
            success3 = send_email(
                subject="🎉 Welcome to ZYPPTS!",
                recipients=['zyppts@gmail.com'],
                template='welcome_email',
                username='TestUser',
                login_url='http://localhost:5000/login'
            )
            
            if success3:
                print("✅ Welcome email sent successfully")
            else:
                print("❌ Failed to send welcome email")
            
            print("\n📧 Test Results Summary:")
            print(f"Basic Test Email: {'✅ Success' if success1 else '❌ Failed'}")
            print(f"New Account Notification: {'✅ Success' if success2 else '❌ Failed'}")
            print(f"Welcome Email: {'✅ Success' if success3 else '❌ Failed'}")
            
            if all([success1, success2, success3]):
                print("\n🎉 All test emails sent successfully!")
                print("Check zyppts@gmail.com for the test emails.")
            else:
                print("\n⚠️  Some test emails failed. Check your email configuration.")
                
    except Exception as e:
        print(f"❌ Error sending test emails: {e}")
        return False
    
    return True

def main():
    """Main function to configure email and send tests"""
    print("🚀 ZYPPTS Email Configuration and Testing")
    print("=" * 60)
    
    # Configure email settings
    if configure_email_settings():
        print("\n" + "=" * 60)
        
        # Ask user if they want to send test emails
        response = input("\nDo you want to send test emails now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            send_test_emails()
        else:
            print("\n📋 To send test emails later:")
            print("1. Set up your Gmail App Password")
            print("2. Update the .env file with your password")
            print("3. Run: python3 scripts/configure_email.py --test")
    
    print("\n📖 Next Steps:")
    print("1. Set up Gmail App Password (see instructions above)")
    print("2. Update .env file with your actual app password")
    print("3. Restart your Flask application")
    print("4. Test the email system through the admin panel")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        send_test_emails()
    else:
        main() 
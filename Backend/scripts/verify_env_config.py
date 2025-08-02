#!/usr/bin/env python3
"""
Verify Environment Configuration
Checks if required environment variables are set without exposing sensitive data
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def verify_env_configuration():
    """Verify that all required environment variables are configured"""
    print("🔍 Verifying Environment Configuration")
    print("=" * 50)
    
    # Load environment variables
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv()
        print("✅ .env file found and loaded")
    else:
        print("❌ .env file not found")
        return False
    
    # Required email configuration
    email_vars = {
        'MAIL_USERNAME': 'Email username (Gmail address)',
        'MAIL_PASSWORD': 'Email password (Gmail app password)',
        'MAIL_DEFAULT_SENDER': 'Default sender email',
        'ADMIN_ALERT_EMAIL': 'Admin alert email address',
        'SITE_URL': 'Site URL for email links'
    }
    
    print("\n📧 Email Configuration:")
    print("-" * 30)
    
    all_configured = True
    for var, description in email_vars.items():
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if 'PASSWORD' in var:
                masked_value = '*' * min(len(value), 8) + '...' if len(value) > 8 else '*' * len(value)
                print(f"✅ {var}: {masked_value} ({description})")
            else:
                print(f"✅ {var}: {value} ({description})")
        else:
            print(f"❌ {var}: Not configured ({description})")
            all_configured = False
    
    # Security configuration
    security_vars = {
        'SECRET_KEY': 'Flask secret key',
        'ADMIN_IP_WHITELIST': 'Admin IP whitelist (optional)',
        'ADMIN_ALLOWED_EMAILS': 'Admin allowed emails (optional)',
        'ADMIN_SECRET_KEY': 'Admin secret key (optional)'
    }
    
    print("\n🔐 Security Configuration:")
    print("-" * 30)
    
    for var, description in security_vars.items():
        value = os.environ.get(var)
        if value:
            if 'SECRET' in var or 'KEY' in var:
                masked_value = '*' * min(len(value), 8) + '...' if len(value) > 8 else '*' * len(value)
                print(f"✅ {var}: {masked_value} ({description})")
            else:
                print(f"✅ {var}: {value} ({description})")
        else:
            if 'optional' in description:
                print(f"⚠️  {var}: Not configured ({description})")
            else:
                print(f"❌ {var}: Not configured ({description})")
                all_configured = False
    
    # Database configuration
    db_vars = {
        'DATABASE_URL': 'Database connection URL',
        'REDIS_HOST': 'Redis host (optional)',
        'REDIS_PORT': 'Redis port (optional)'
    }
    
    print("\n🗄️ Database Configuration:")
    print("-" * 30)
    
    for var, description in db_vars.items():
        value = os.environ.get(var)
        if value:
            if 'URL' in var and '://' in value:
                # Mask database password in URL
                if '@' in value:
                    parts = value.split('@')
                    if len(parts) == 2:
                        auth_part = parts[0]
                        if ':' in auth_part:
                            user_pass = auth_part.split(':')
                            if len(user_pass) >= 3:  # user:pass@host format
                                masked_url = f"{user_pass[0]}:****@{parts[1]}"
                                print(f"✅ {var}: {masked_url} ({description})")
                            else:
                                print(f"✅ {var}: {value} ({description})")
                        else:
                            print(f"✅ {var}: {value} ({description})")
                    else:
                        print(f"✅ {var}: {value} ({description})")
                else:
                    print(f"✅ {var}: {value} ({description})")
            else:
                print(f"✅ {var}: {value} ({description})")
        else:
            if 'optional' in description:
                print(f"⚠️  {var}: Not configured ({description})")
            else:
                print(f"❌ {var}: Not configured ({description})")
                all_configured = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Configuration Summary:")
    print("=" * 50)
    
    if all_configured:
        print("✅ All required environment variables are configured!")
        print("🎉 Your email notification system should be working properly.")
    else:
        print("❌ Some required environment variables are missing.")
        print("📝 Please check your .env file and ensure all required variables are set.")
    
    return all_configured

def test_email_configuration():
    """Test if email configuration is working"""
    print("\n🧪 Testing Email Configuration:")
    print("-" * 30)
    
    try:
        from flask import Flask
        from flask_mail import Mail
        
        # Create a minimal Flask app for testing
        app = Flask(__name__)
        app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
        app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
        app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
        app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
        app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
        app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
        
        mail = Mail(app)
        
        with app.app_context():
            # Test mail configuration
            if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
                print("✅ Email configuration appears valid")
                print(f"📧 Server: {app.config['MAIL_SERVER']}:{app.config['MAIL_PORT']}")
                print(f"📧 Username: {app.config['MAIL_USERNAME']}")
                print(f"📧 Sender: {app.config['MAIL_DEFAULT_SENDER']}")
                print(f"📧 Admin Alert: {os.environ.get('ADMIN_ALERT_EMAIL', 'Not configured')}")
                return True
            else:
                print("❌ Email credentials not configured")
                return False
                
    except Exception as e:
        print(f"❌ Error testing email configuration: {e}")
        return False

def main():
    """Main function"""
    print("🚀 ZYPPTS Environment Configuration Verification")
    print("=" * 60)
    
    # Verify configuration
    config_ok = verify_env_configuration()
    
    if config_ok:
        # Test email configuration
        email_ok = test_email_configuration()
        
        if email_ok:
            print("\n🎉 Configuration verification complete!")
            print("✅ Your email notification system should be working.")
            print("\n📧 To test email functionality:")
            print("1. Go to Admin Panel: http://127.0.0.1:5003/admin/")
            print("2. Navigate to Notifications section")
            print("3. Click 'Send Test Email'")
        else:
            print("\n⚠️  Email configuration needs attention.")
            print("📝 Check your Gmail app password and settings.")
    else:
        print("\n❌ Configuration verification failed.")
        print("📝 Please update your .env file with missing variables.")

if __name__ == "__main__":
    main() 
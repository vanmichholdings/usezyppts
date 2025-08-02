#!/usr/bin/env python3
"""
Test Email Sender Name
"""

import os
import sys
from pathlib import Path

def test_email_sender():
    """Test email sender name configuration"""
    print("🧪 Testing Email Sender Name")
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
            # Check current configuration
            print("📧 Current Email Configuration:")
            print(f"  MAIL_DEFAULT_SENDER: {app.config.get('MAIL_DEFAULT_SENDER')}")
            print(f"  MAIL_USERNAME: {app.config.get('MAIL_USERNAME')}")
            print(f"  ADMIN_ALERT_EMAIL: {app.config.get('ADMIN_ALERT_EMAIL')}")
            
            # Send test email
            print("\n📤 Sending test email...")
            success = send_email(
                subject="🧪 Sender Name Test - Zyppts",
                recipients=[app.config.get('ADMIN_ALERT_EMAIL', 'zyppts@gmail.com')],
                template='test_notification',
                admin_name='Sender Test',
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            )
            
            if success:
                print("✅ Test email sent successfully")
                print("\n📋 Check your email inbox:")
                print(f"  - Look for: '🧪 Sender Name Test - Zyppts'")
                print(f"  - Sender should show: 'Zyppts HQ' instead of 'zyppts@gmail.com'")
                print(f"  - Logo should display as text instead of being blocked")
            else:
                print("❌ Failed to send test email")
                
    except Exception as e:
        print(f"❌ Error testing email sender: {e}")
        return False
    
    return True

def test_daily_summary():
    """Test daily summary email"""
    print("\n📊 Testing Daily Summary Email")
    print("=" * 60)
    
    try:
        from app_config import create_app
        from utils.email_notifications import send_daily_summary
        
        app = create_app()
        
        with app.app_context():
            success = send_daily_summary()
            
            if success:
                print("✅ Daily summary sent successfully")
                print("📋 Check your email for the daily summary with proper branding")
            else:
                print("❌ Failed to send daily summary")
                
    except Exception as e:
        print(f"❌ Error testing daily summary: {e}")
        return False
    
    return True

def test_weekly_report():
    """Test weekly report email"""
    print("\n📈 Testing Weekly Report Email")
    print("=" * 60)
    
    try:
        from app_config import create_app
        from utils.email_notifications import send_weekly_report
        
        app = create_app()
        
        with app.app_context():
            success = send_weekly_report()
            
            if success:
                print("✅ Weekly report sent successfully")
                print("📋 Check your email for the weekly report with proper branding")
            else:
                print("❌ Failed to send weekly report")
                
    except Exception as e:
        print(f"❌ Error testing weekly report: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Email Sender Name and Branding Test")
    print("=" * 60)
    
    # Test sender name
    test_email_sender()
    
    print("\n" + "=" * 60)
    
    # Test daily summary
    test_daily_summary()
    
    print("\n" + "=" * 60)
    
    # Test weekly report
    test_weekly_report()
    
    print("\n📖 Summary:")
    print("1. Check your email inbox for all test emails")
    print("2. Verify sender name shows 'Zyppts HQ' instead of email address")
    print("3. Verify logo displays as text (not blocked as remote content)")
    print("4. Verify all emails have consistent branding")

if __name__ == "__main__":
    main() 
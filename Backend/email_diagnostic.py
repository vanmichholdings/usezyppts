#!/usr/bin/env python3
"""
Email Notification Diagnostic Tool
Tests and troubleshoots email notification system
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from flask import Flask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_app():
    """Create a minimal Flask app for testing"""
    app = Flask(__name__)
    
    # Load configuration
    from config import Config
    app.config.from_object(Config)
    
    # Initialize Flask extensions
    from flask_mail import Mail
    mail = Mail()
    mail.init_app(app)
    
    return app

def test_email_configuration():
    """Test email configuration"""
    print("=" * 60)
    print("EMAIL CONFIGURATION TEST")
    print("=" * 60)
    
    app = create_test_app()
    
    with app.app_context():
        # Check email settings
        print(f"MAIL_SERVER: {app.config.get('MAIL_SERVER')}")
        print(f"MAIL_PORT: {app.config.get('MAIL_PORT')}")
        print(f"MAIL_USE_TLS: {app.config.get('MAIL_USE_TLS')}")
        print(f"MAIL_USERNAME: {app.config.get('MAIL_USERNAME')}")
        print(f"MAIL_PASSWORD: {'***SET***' if app.config.get('MAIL_PASSWORD') else 'NOT SET'}")
        print(f"MAIL_DEFAULT_SENDER: {app.config.get('MAIL_DEFAULT_SENDER')}")
        print(f"ADMIN_ALERT_EMAIL: {app.config.get('ADMIN_ALERT_EMAIL')}")
        
        # Check if required settings are configured
        missing_settings = []
        if not app.config.get('MAIL_USERNAME'):
            missing_settings.append('MAIL_USERNAME')
        if not app.config.get('MAIL_PASSWORD'):
            missing_settings.append('MAIL_PASSWORD')
        if not app.config.get('ADMIN_ALERT_EMAIL'):
            missing_settings.append('ADMIN_ALERT_EMAIL')
        
        if missing_settings:
            print(f"\n❌ Missing email settings: {', '.join(missing_settings)}")
            return False
        else:
            print("\n✅ All email settings are configured")
            return True

def test_email_connection():
    """Test email server connection"""
    print("\n" + "=" * 60)
    print("EMAIL CONNECTION TEST")
    print("=" * 60)
    
    app = create_test_app()
    
    with app.app_context():
        try:
            from flask_mail import Mail
            mail = Mail()
            mail.init_app(app)
            
            # Test connection
            with mail.connect() as conn:
                print("✅ Email server connection successful")
                return True
                
        except Exception as e:
            print(f"❌ Email server connection failed: {e}")
            return False

def test_daily_summary_function():
    """Test daily summary function"""
    print("\n" + "=" * 60)
    print("DAILY SUMMARY FUNCTION TEST")
    print("=" * 60)
    
    app = create_test_app()
    
    with app.app_context():
        try:
            from utils.email_notifications import send_daily_summary
            
            # Test the function
            result = send_daily_summary()
            
            if result:
                print("✅ Daily summary function executed successfully")
                return True
            else:
                print("❌ Daily summary function returned False")
                return False
                
        except Exception as e:
            print(f"❌ Daily summary function failed: {e}")
            return False

def test_scheduler_status():
    """Test scheduler status"""
    print("\n" + "=" * 60)
    print("SCHEDULER STATUS TEST")
    print("=" * 60)
    
    try:
        from utils.scheduled_tasks import scheduler, get_scheduled_jobs
        
        if scheduler and scheduler.running:
            print("✅ Scheduler is running")
            
            # Get scheduled jobs
            jobs = get_scheduled_jobs()
            print(f"📅 Scheduled jobs: {len(jobs)}")
            
            for job in jobs:
                print(f"  - {job.name} (ID: {job.id})")
                print(f"    Next run: {job.next_run_time}")
                print(f"    Trigger: {job.trigger}")
            
            return True
        else:
            print("❌ Scheduler is not running")
            return False
            
    except Exception as e:
        print(f"❌ Scheduler test failed: {e}")
        return False

def test_email_template():
    """Test email template rendering"""
    print("\n" + "=" * 60)
    print("EMAIL TEMPLATE TEST")
    print("=" * 60)
    
    app = create_test_app()
    
    with app.app_context():
        try:
            from flask import render_template
            
            # Test daily summary template
            template_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'new_users': 5,
                'new_subscriptions': 2,
                'new_uploads': 15,
                'top_users': [],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
            html_content = render_template('emails/daily_summary.html', **template_data)
            
            if html_content and len(html_content) > 100:
                print("✅ Daily summary template renders successfully")
                print(f"   Template size: {len(html_content)} characters")
                return True
            else:
                print("❌ Daily summary template rendering failed")
                return False
                
        except Exception as e:
            print(f"❌ Template test failed: {e}")
            return False

def send_test_email():
    """Send a test email"""
    print("\n" + "=" * 60)
    print("TEST EMAIL SEND")
    print("=" * 60)
    
    app = create_test_app()
    
    with app.app_context():
        try:
            from utils.email_notifications import send_email
            
            # Send test email
            result = send_email(
                subject="🧪 Email System Test - Zyppts",
                recipients=[app.config.get('ADMIN_ALERT_EMAIL')],
                template='test_notification',
                test_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                system_status="Email system is working correctly"
            )
            
            if result:
                print("✅ Test email sent successfully")
                print(f"   Sent to: {app.config.get('ADMIN_ALERT_EMAIL')}")
                return True
            else:
                print("❌ Test email failed to send")
                return False
                
        except Exception as e:
            print(f"❌ Test email failed: {e}")
            return False

def check_environment_variables():
    """Check environment variables"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 60)
    
    required_vars = [
        'MAIL_USERNAME',
        'MAIL_PASSWORD',
        'ADMIN_ALERT_EMAIL'
    ]
    
    optional_vars = [
        'MAIL_DEFAULT_SENDER',
        'SITE_URL',
        'FLASK_ENV'
    ]
    
    print("Required variables:")
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {'***SET***' if 'PASSWORD' in var else value}")
        else:
            print(f"  ❌ {var}: NOT SET")
    
    print("\nOptional variables:")
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️  {var}: NOT SET (using default)")

def run_manual_daily_summary():
    """Manually trigger daily summary"""
    print("\n" + "=" * 60)
    print("MANUAL DAILY SUMMARY TRIGGER")
    print("=" * 60)
    
    try:
        from utils.scheduled_tasks import run_manual_daily_summary
        
        result = run_manual_daily_summary()
        
        if result:
            print("✅ Manual daily summary triggered successfully")
            return True
        else:
            print("❌ Manual daily summary failed")
            return False
            
    except Exception as e:
        print(f"❌ Manual daily summary failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🧪 ZYPPTS EMAIL DIAGNOSTIC TOOL")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", check_environment_variables),
        ("Email Configuration", test_email_configuration),
        ("Email Connection", test_email_connection),
        ("Email Template", test_email_template),
        ("Scheduler Status", test_scheduler_status),
        ("Daily Summary Function", test_daily_summary_function),
        ("Test Email Send", send_test_email),
        ("Manual Daily Summary", run_manual_daily_summary)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Email system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        
        # Provide recommendations
        print("\n🔧 RECOMMENDATIONS:")
        if not any(name == "Email Configuration" and result for name, result in results):
            print("- Configure email settings (MAIL_USERNAME, MAIL_PASSWORD, ADMIN_ALERT_EMAIL)")
        if not any(name == "Email Connection" and result for name, result in results):
            print("- Check email server settings and credentials")
        if not any(name == "Scheduler Status" and result for name, result in results):
            print("- Restart the application to initialize the scheduler")
        if not any(name == "Daily Summary Function" and result for name, result in results):
            print("- Check database connection and models")

if __name__ == "__main__":
    main() 
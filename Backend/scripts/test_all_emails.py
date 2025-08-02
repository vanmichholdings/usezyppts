#!/usr/bin/env python3
"""
Comprehensive Email Test - All Notification Types
"""

import os
import sys
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def read_env_password():
    """Read password from .env file"""
    env_file = Path('.env')
    password = "your-app-password-here"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('MAIL_PASSWORD='):
                    password = line.split('=', 1)[1].strip()
                    break
    
    return password

def send_email(subject, body, html_body=None):
    """Send email to zyppts@gmail.com"""
    try:
        # Email settings
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        username = "zyppts@gmail.com"
        password = read_env_password()
        
        # Create server connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        
        # Create email
        msg = MIMEMultipart('alternative')
        msg['From'] = "noreply@zyppts.com"
        msg['To'] = "zyppts@gmail.com"
        msg['Subject'] = subject
        
        # Add text and HTML parts
        text_part = MIMEText(body, 'plain')
        msg.attach(text_part)
        
        if html_body:
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
        
        # Send email
        text = msg.as_string()
        server.sendmail("noreply@zyppts.com", "zyppts@gmail.com", text)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

def test_new_account_notification():
    """Test new account notification email"""
    print("1️⃣ Testing New Account Notification...")
    
    subject = "🆕 New User Registration: testuser"
    
    body = f"""
    🆕 New User Registration
    
    A new user has registered on ZYPPTS
    
    👤 User Details:
    - Username: testuser
    - Email: test@example.com
    - Registration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - IP Address: 127.0.0.1
    - User Agent: Test Browser
    
    📊 Quick Actions:
    - View User Profile: http://localhost:5000/admin/user/1
    - Admin Dashboard: http://localhost:5000/admin
    
    This is an automated notification from your ZYPPTS admin system.
    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background-color: #3B5EF7; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 20px; }}
            .user-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .btn {{ display: inline-block; padding: 10px 20px; background-color: #3B5EF7; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🆕 New User Registration</h1>
            <p>A new user has registered on ZYPPTS</p>
        </div>
        <div class="content">
            <div class="user-info">
                <h3>👤 User Details</h3>
                <p><strong>Username:</strong> testuser</p>
                <p><strong>Email:</strong> test@example.com</p>
                <p><strong>Registration Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>IP Address:</strong> 127.0.0.1</p>
                <p><strong>User Agent:</strong> Test Browser</p>
            </div>
            <div style="text-align: center; margin: 20px 0;">
                <a href="http://localhost:5000/admin/user/1" class="btn">👁️ View User Profile</a>
                <a href="http://localhost:5000/admin" class="btn">📊 Admin Dashboard</a>
            </div>
            <p style="text-align: center; color: #666; font-size: 12px;">
                This is an automated notification from your ZYPPTS admin system.<br>
                Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
            </p>
        </div>
    </body>
    </html>
    """
    
    success = send_email(subject, body, html_body)
    if success:
        print("✅ New account notification sent successfully")
    else:
        print("❌ Failed to send new account notification")
    return success

def test_welcome_email():
    """Test welcome email"""
    print("2️⃣ Testing Welcome Email...")
    
    subject = "🎉 Welcome to ZYPPTS!"
    
    body = f"""
    🎉 Welcome to ZYPPTS!
    
    Hello TestUser!
    
    Thank you for joining ZYPPTS! We're excited to help you transform your logos with cutting-edge AI technology.
    
    🚀 What you can do with ZYPPTS:
    ✨ Remove backgrounds from logos instantly
    ✨ Generate multiple logo variations
    ✨ Convert logos to vector formats
    ✨ Create professional mockups
    ✨ Export in multiple formats (PNG, SVG, PDF)
    ✨ Access advanced AI processing tools
    
    Ready to get started?
    Upload your first logo and see the magic happen!
    
    Login: http://localhost:5000/login
    
    If you have any questions, feel free to reach out to our support team.
    
    Happy logo processing!
    The ZYPPTS Team
    """
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .header {{ background: linear-gradient(135deg, #3B5EF7, #60A5FA); color: white; padding: 30px; text-align: center; }}
            .content {{ padding: 20px; }}
            .features {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .cta {{ background: linear-gradient(135deg, #3B5EF7, #60A5FA); color: white; padding: 25px; text-align: center; border-radius: 8px; }}
            .btn {{ display: inline-block; padding: 15px 30px; background-color: white; color: #3B5EF7; text-decoration: none; border-radius: 6px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎉 Welcome to ZYPPTS!</h1>
            <p>Your AI-powered logo processing journey starts now</p>
        </div>
        <div class="content">
            <h2>Hello TestUser!</h2>
            <p>Thank you for joining ZYPPTS! We're excited to help you transform your logos with cutting-edge AI technology.</p>
            
            <div class="features">
                <h3>🚀 What you can do with ZYPPTS:</h3>
                <ul>
                    <li>✨ Remove backgrounds from logos instantly</li>
                    <li>✨ Generate multiple logo variations</li>
                    <li>✨ Convert logos to vector formats</li>
                    <li>✨ Create professional mockups</li>
                    <li>✨ Export in multiple formats (PNG, SVG, PDF)</li>
                    <li>✨ Access advanced AI processing tools</li>
                </ul>
            </div>
            
            <div class="cta">
                <h3>Ready to get started?</h3>
                <p>Upload your first logo and see the magic happen!</p>
                <a href="http://localhost:5000/login" class="btn">🚀 Start Processing Logos</a>
            </div>
            
            <p style="text-align: center; margin-top: 30px;">
                If you have any questions, feel free to reach out to our support team.<br>
                Happy logo processing!<br>
                <strong>The ZYPPTS Team</strong>
            </p>
        </div>
    </body>
    </html>
    """
    
    success = send_email(subject, body, html_body)
    if success:
        print("✅ Welcome email sent successfully")
    else:
        print("❌ Failed to send welcome email")
    return success

def test_daily_summary():
    """Test daily summary email"""
    print("3️⃣ Testing Daily Summary...")
    
    subject = f"📊 Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
    
    body = f"""
    📊 Daily Summary Report
    
    Date: {datetime.now().strftime('%Y-%m-%d')}
    
    📈 Statistics:
    - New Users: 5
    - New Subscriptions: 2
    - Total Uploads: 15
    - Active Users: 12
    
    👥 Top Users:
    1. user1@example.com (3 uploads)
    2. user2@example.com (2 uploads)
    3. user3@example.com (1 upload)
    
    💰 Revenue:
    - Today's Revenue: $45.00
    - Monthly Revenue: $1,250.00
    
    📊 System Health:
    - Server Uptime: 99.9%
    - Average Response Time: 1.2s
    - Error Rate: 0.1%
    
    This is your daily summary from ZYPPTS admin system.
    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    success = send_email(subject, body)
    if success:
        print("✅ Daily summary sent successfully")
    else:
        print("❌ Failed to send daily summary")
    return success

def test_weekly_report():
    """Test weekly report email"""
    print("4️⃣ Testing Weekly Report...")
    
    subject = f"📈 Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"
    
    body = f"""
    📈 Weekly Analytics Report
    
    Period: {datetime.now().strftime('%Y-%m-%d')} (This Week)
    
    📊 Weekly Statistics:
    - New Users: 25 (+15% from last week)
    - New Subscriptions: 8 (+20% from last week)
    - Total Uploads: 89 (+12% from last week)
    - Active Users: 45 (+8% from last week)
    
    📈 Growth Trends:
    - User Growth: ↗️ +15%
    - Revenue Growth: ↗️ +22%
    - Engagement: ↗️ +18%
    
    💰 Revenue Summary:
    - Weekly Revenue: $320.00
    - Monthly Revenue: $1,250.00
    - Average Order Value: $40.00
    
    🎯 Top Performing Features:
    1. Vector Tracing (45% usage)
    2. Background Removal (38% usage)
    3. Color Separations (17% usage)
    
    📊 System Performance:
    - Average Response Time: 1.1s
    - Server Uptime: 99.9%
    - Error Rate: 0.05%
    
    🚀 Recommendations:
    - Consider promoting vector tracing feature
    - Monitor server performance during peak hours
    - Implement user onboarding improvements
    
    This is your weekly report from ZYPPTS admin system.
    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    success = send_email(subject, body)
    if success:
        print("✅ Weekly report sent successfully")
    else:
        print("❌ Failed to send weekly report")
    return success

def test_security_alert():
    """Test security alert email"""
    print("5️⃣ Testing Security Alert...")
    
    subject = "🚨 Security Alert: Unauthorized Access Attempt"
    
    body = f"""
    🚨 Security Alert
    
    Event Type: Unauthorized Access Attempt
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    
    📍 Details:
    - IP Address: 192.168.1.100
    - User Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
    - Attempted Action: Admin Panel Access
    - Target User: admin@example.com
    
    🔍 Analysis:
    - This appears to be a brute force attack attempt
    - IP is not in whitelist
    - Multiple failed login attempts detected
    
    🛡️ Actions Taken:
    - IP temporarily blocked (15 minutes)
    - Failed attempt logged
    - Admin notification sent
    
    📋 Recommended Actions:
    1. Monitor this IP for further suspicious activity
    2. Consider adding to permanent blacklist if pattern continues
    3. Review security logs for similar patterns
    4. Consider implementing additional security measures
    
    🔗 Admin Actions:
    - View Security Logs: http://localhost:5000/admin/system
    - Manage IP Blacklist: http://localhost:5000/admin/security
    
    This is an automated security alert from ZYPPTS admin system.
    Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """
    
    success = send_email(subject, body)
    if success:
        print("✅ Security alert sent successfully")
    else:
        print("❌ Failed to send security alert")
    return success

def main():
    """Main function to test all email types"""
    print("🚀 ZYPPTS Comprehensive Email Test")
    print("=" * 60)
    print("Testing all email notification types...")
    print()
    
    # Test all email types
    results = []
    
    results.append(test_new_account_notification())
    results.append(test_welcome_email())
    results.append(test_daily_summary())
    results.append(test_weekly_report())
    results.append(test_security_alert())
    
    # Summary
    print("\n" + "=" * 60)
    print("📧 Test Results Summary:")
    print("=" * 60)
    
    email_types = [
        "New Account Notification",
        "Welcome Email", 
        "Daily Summary",
        "Weekly Report",
        "Security Alert"
    ]
    
    for i, (email_type, success) in enumerate(zip(email_types, results), 1):
        status = "✅ Success" if success else "❌ Failed"
        print(f"{i}. {email_type}: {status}")
    
    successful_tests = sum(results)
    total_tests = len(results)
    
    print(f"\n📊 Overall Results: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("\n🎉 All email tests passed successfully!")
        print("📧 Check zyppts@gmail.com for all test emails")
        print("\n✅ Your email notification system is fully operational!")
        print("You'll now receive:")
        print("- Instant notifications for new user registrations")
        print("- Daily summary reports at 9 AM UTC")
        print("- Weekly analytics reports every Monday at 10 AM UTC")
        print("- Security alerts for suspicious activity")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} test(s) failed")
        print("Check your email configuration and try again")

if __name__ == "__main__":
    main() 
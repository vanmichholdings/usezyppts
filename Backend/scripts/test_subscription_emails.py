#!/usr/bin/env python3
"""
Test script for subscription email templates
This script sends test emails for all subscription-related templates.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_subscription_emails():
    """
    Test all subscription email templates
    """
    print("🧪 Testing Subscription Email Templates")
    print("=" * 50)
    print()
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        from utils.email_notifications import (
            send_payment_confirmation,
            send_payment_failed,
            send_subscription_upgrade,
            send_account_cancellation
        )
        
        # Create Flask app context
        app = create_app()
        with app.app_context():
            
            # Test email address
            test_email = input("Enter test email address: ").strip()
            if not test_email:
                print("❌ No email address provided")
                return False
            
            # Create a test user
            test_user = User(
                username='test_user',
                email=test_email,
                created_at=datetime.utcnow()
            )
            
            # Create a test subscription
            test_subscription = Subscription(
                plan='pro',
                status='active',
                monthly_credits=100,
                used_credits=25,
                start_date=datetime.utcnow() - timedelta(days=30),
                billing_cycle='monthly',
                payment_id='sub_test_123456'
            )
            
            print("📧 Sending test emails...")
            print()
            
            # Test 1: Payment Confirmation
            print("1️⃣ Testing Payment Confirmation Email...")
            success = send_payment_confirmation(
                user=test_user,
                subscription=test_subscription,
                amount=9.99,
                transaction_id='pi_test_123456',
                payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            )
            if success:
                print("   ✅ Payment confirmation email sent")
            else:
                print("   ❌ Failed to send payment confirmation email")
            print()
            
            # Test 2: Payment Failed
            print("2️⃣ Testing Payment Failed Email...")
            success = send_payment_failed(
                user=test_user,
                subscription=test_subscription,
                amount=9.99,
                transaction_id='pi_test_789012',
                payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                error_message='Your card was declined. Please update your payment method.'
            )
            if success:
                print("   ✅ Payment failed email sent")
            else:
                print("   ❌ Failed to send payment failed email")
            print()
            
            # Test 3: Subscription Upgrade
            print("3️⃣ Testing Subscription Upgrade Email...")
            new_features = [
                '15 Team Members (up from 5)',
                'Premium Analytics',
                '24/7 Priority Support',
                'API Access'
            ]
            success = send_subscription_upgrade(
                user=test_user,
                old_plan='pro',
                new_plan='studio',
                old_price=9.99,
                new_price=29.99,
                billing_cycle='monthly',
                effective_date=datetime.utcnow().strftime('%Y-%m-%d'),
                next_billing_date=(datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d'),
                new_features=new_features
            )
            if success:
                print("   ✅ Subscription upgrade email sent")
            else:
                print("   ❌ Failed to send subscription upgrade email")
            print()
            
            # Test 4: Account Cancellation
            print("4️⃣ Testing Account Cancellation Email...")
            success = send_account_cancellation(
                user=test_user,
                subscription=test_subscription,
                cancellation_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                access_until_date=(datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d'),
                cancellation_reason='Switching to a different service'
            )
            if success:
                print("   ✅ Account cancellation email sent")
            else:
                print("   ❌ Failed to send account cancellation email")
            print()
            
            print("🎉 All test emails completed!")
            print()
            print("📋 Summary:")
            print("   • Payment Confirmation: ✅")
            print("   • Payment Failed: ✅")
            print("   • Subscription Upgrade: ✅")
            print("   • Account Cancellation: ✅")
            print()
            print(f"📧 Check your email at: {test_email}")
            print("   (Check spam folder if not received)")
            
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the Backend directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_email():
    """
    Test a single email template
    """
    print("🧪 Test Single Email Template")
    print("=" * 40)
    print()
    
    print("Available templates:")
    print("1. Payment Confirmation")
    print("2. Payment Failed")
    print("3. Subscription Upgrade")
    print("4. Account Cancellation")
    print()
    
    choice = input("Select template (1-4): ").strip()
    test_email = input("Enter test email address: ").strip()
    
    if not test_email:
        print("❌ No email address provided")
        return False
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        from utils.email_notifications import (
            send_payment_confirmation,
            send_payment_failed,
            send_subscription_upgrade,
            send_account_cancellation
        )
        
        # Create Flask app context
        app = create_app()
        with app.app_context():
            
            # Create test data
            test_user = User(
                username='test_user',
                email=test_email,
                created_at=datetime.utcnow()
            )
            
            test_subscription = Subscription(
                plan='pro',
                status='active',
                monthly_credits=100,
                used_credits=25,
                start_date=datetime.utcnow() - timedelta(days=30),
                billing_cycle='monthly',
                payment_id='sub_test_123456'
            )
            
            success = False
            
            if choice == '1':
                print("📧 Sending Payment Confirmation Email...")
                success = send_payment_confirmation(
                    user=test_user,
                    subscription=test_subscription,
                    amount=9.99,
                    transaction_id='pi_test_123456',
                    payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )
            elif choice == '2':
                print("📧 Sending Payment Failed Email...")
                success = send_payment_failed(
                    user=test_user,
                    subscription=test_subscription,
                    amount=9.99,
                    transaction_id='pi_test_789012',
                    payment_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    error_message='Your card was declined. Please update your payment method.'
                )
            elif choice == '3':
                print("📧 Sending Subscription Upgrade Email...")
                new_features = [
                    '15 Team Members (up from 5)',
                    'Premium Analytics',
                    '24/7 Priority Support',
                    'API Access'
                ]
                success = send_subscription_upgrade(
                    user=test_user,
                    old_plan='pro',
                    new_plan='studio',
                    old_price=9.99,
                    new_price=29.99,
                    billing_cycle='monthly',
                    effective_date=datetime.utcnow().strftime('%Y-%m-%d'),
                    next_billing_date=(datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    new_features=new_features
                )
            elif choice == '4':
                print("📧 Sending Account Cancellation Email...")
                success = send_account_cancellation(
                    user=test_user,
                    subscription=test_subscription,
                    cancellation_date=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    access_until_date=(datetime.utcnow() + timedelta(days=30)).strftime('%Y-%m-%d'),
                    cancellation_reason='Switching to a different service'
                )
            else:
                print("❌ Invalid choice")
                return False
            
            if success:
                print("✅ Email sent successfully!")
                print(f"📧 Check your email at: {test_email}")
            else:
                print("❌ Failed to send email")
            
            return success
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test subscription email templates")
    parser.add_argument("--all", action="store_true", help="Test all email templates")
    parser.add_argument("--single", action="store_true", help="Test a single email template")
    
    args = parser.parse_args()
    
    if args.all:
        test_subscription_emails()
    elif args.single:
        test_single_email()
    else:
        print("Please specify an option:")
        print("  --all    : Test all email templates")
        print("  --single : Test a single email template")
        print()
        print("Example: python test_subscription_emails.py --all") 
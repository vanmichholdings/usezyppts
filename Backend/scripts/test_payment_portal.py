#!/usr/bin/env python3
"""
Test script for payment portal functionality
This script tests the Stripe Customer Portal integration.
"""

import os
import sys

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_stripe_config():
    """
    Check if Stripe is properly configured
    """
    print("🔍 Checking Stripe Configuration...")
    print()
    
    try:
        from config import Config
        
        if Config.STRIPE_SECRET_KEY:
            print("✅ Stripe Secret Key: Configured")
            print(f"   Key starts with: {Config.STRIPE_SECRET_KEY[:10]}...")
        else:
            print("❌ Stripe Secret Key: Not configured")
            
        if Config.STRIPE_PUBLISHABLE_KEY:
            print("✅ Stripe Publishable Key: Configured")
            print(f"   Key starts with: {Config.STRIPE_PUBLISHABLE_KEY[:10]}...")
        else:
            print("❌ Stripe Publishable Key: Not configured")
            
        if Config.STRIPE_WEBHOOK_SECRET:
            print("✅ Stripe Webhook Secret: Configured")
            print(f"   Secret starts with: {Config.STRIPE_WEBHOOK_SECRET[:10]}...")
        else:
            print("❌ Stripe Webhook Secret: Not configured")
            
    except ImportError as e:
        print(f"❌ Could not import config: {e}")
    except Exception as e:
        print(f"❌ Error checking config: {e}")
    
    print()

def test_payment_portal():
    """
    Test the payment portal functionality
    """
    print("🧪 Testing Payment Portal Functionality")
    print("=" * 50)
    print()
    
    print("📋 Manual Testing Steps:")
    print("1. Start the Flask application: cd Backend && python app.py")
    print("2. Log in to your account")
    print("3. Go to the Account page")
    print("4. Click 'Change Payment Method' button")
    print("5. Verify you're redirected to Stripe Customer Portal")
    print("6. Test updating payment method in Stripe")
    print("7. Verify return to account page with success message")
    
    print()
    print("🔧 Configuration Requirements:")
    print("• Stripe Customer Portal must be enabled in your Stripe Dashboard")
    print("• Webhook endpoint configured for portal events")
    print("• Valid Stripe API keys configured")
    
    print()
    print("📋 Stripe Dashboard Setup:")
    print("1. Go to Stripe Dashboard → Settings → Customer Portal")
    print("2. Enable Customer Portal feature")
    print("3. Configure business information and branding")
    print("4. Set up return URLs")
    print("5. Configure allowed features (payment methods, invoices, etc.)")
    
    return True

if __name__ == "__main__":
    check_stripe_config()
    test_payment_portal() 
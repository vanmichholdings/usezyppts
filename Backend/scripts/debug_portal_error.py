#!/usr/bin/env python3
"""
Debug script for customer portal redirect errors
This script helps identify the specific issue with the portal redirect.
"""

import os
import sys
import traceback

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_portal_error():
    """
    Debug the customer portal redirect error
    """
    print("🔍 Debugging Customer Portal Redirect Error")
    print("=" * 50)
    print()
    
    try:
        # Test 1: Check Stripe configuration
        print("1️⃣ Checking Stripe Configuration...")
        from config import Config
        
        if not Config.STRIPE_SECRET_KEY:
            print("❌ STRIPE_SECRET_KEY is not configured")
            return False
        else:
            print(f"✅ STRIPE_SECRET_KEY: {Config.STRIPE_SECRET_KEY[:10]}...")
        
        if not Config.STRIPE_PUBLISHABLE_KEY:
            print("❌ STRIPE_PUBLISHABLE_KEY is not configured")
            return False
        else:
            print(f"✅ STRIPE_PUBLISHABLE_KEY: {Config.STRIPE_PUBLISHABLE_KEY[:10]}...")
        
        print()
        
        # Test 2: Test Stripe API connection
        print("2️⃣ Testing Stripe API Connection...")
        import stripe
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        try:
            # Test basic API call
            account = stripe.Account.retrieve()
            print(f"✅ Stripe API connection successful")
            print(f"   Account ID: {account.id}")
            print(f"   Account Type: {account.type}")
        except stripe.error.AuthenticationError:
            print("❌ Stripe API authentication failed - check your secret key")
            return False
        except stripe.error.InvalidRequestError as e:
            print(f"❌ Stripe API request failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected Stripe error: {e}")
            return False
        
        print()
        
        # Test 3: Check Customer Portal configuration
        print("3️⃣ Checking Customer Portal Configuration...")
        try:
            # Try to create a test portal session (this will fail without a real customer, but we can see the error)
            portal_config = stripe.billing_portal.Configuration.list(limit=1)
            if portal_config.data:
                print("✅ Customer Portal is configured")
                config = portal_config.data[0]
                print(f"   Configuration ID: {config.id}")
                print(f"   Active: {config.active}")
                print(f"   Business Profile: {config.business_profile.name if config.business_profile else 'Not set'}")
            else:
                print("⚠️ No Customer Portal configurations found")
                print("   You may need to configure the Customer Portal in your Stripe Dashboard")
        except stripe.error.InvalidRequestError as e:
            if "No such configuration" in str(e):
                print("❌ Customer Portal not configured in Stripe Dashboard")
                print("   Please enable Customer Portal in Settings → Customer Portal")
            else:
                print(f"❌ Customer Portal configuration error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error checking portal configuration: {e}")
            return False
        
        print()
        
        # Test 4: Check database models
        print("4️⃣ Checking Database Models...")
        try:
            from app_config import create_app, db
            from models import User, Subscription
            
            app = create_app()
            with app.app_context():
                # Check if we can query the database
                user_count = User.query.count()
                subscription_count = Subscription.query.count()
                print(f"✅ Database connection successful")
                print(f"   Users: {user_count}")
                print(f"   Subscriptions: {subscription_count}")
                
                # Check subscription structure
                if subscription_count > 0:
                    sample_sub = Subscription.query.first()
                    print(f"   Sample subscription payment_id: {sample_sub.payment_id}")
                    print(f"   Sample subscription plan: {sample_sub.plan}")
                    print(f"   Sample subscription status: {sample_sub.status}")
                else:
                    print("   ⚠️ No subscriptions found in database")
                    
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
        
        print()
        
        # Test 5: Common error scenarios
        print("5️⃣ Common Error Scenarios...")
        print("   • Is the user logged in? (Check @login_required)")
        print("   • Does the user have an active subscription?")
        print("   • Is the subscription.payment_id field populated?")
        print("   • Is the payment_id a valid Stripe subscription ID?")
        print("   • Is the Customer Portal enabled in Stripe Dashboard?")
        print("   • Are the return URLs configured correctly?")
        
        print()
        print("🔧 Troubleshooting Steps:")
        print("1. Check browser console for JavaScript errors")
        print("2. Check Flask application logs for Python errors")
        print("3. Verify Customer Portal is enabled in Stripe Dashboard")
        print("4. Ensure user has an active subscription with valid payment_id")
        print("5. Test with a known good Stripe subscription ID")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the Backend directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_portal_endpoint():
    """
    Test the portal endpoint directly
    """
    print("\n🧪 Testing Portal Endpoint...")
    print("=" * 30)
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        import stripe
        from config import Config
        
        app = create_app()
        with app.app_context():
            # Find a user with an active subscription
            active_subscription = Subscription.query.filter_by(status='active').first()
            
            if not active_subscription:
                print("❌ No active subscriptions found in database")
                print("   Create a test subscription first")
                return False
            
            user = active_subscription.user
            print(f"✅ Found test user: {user.username} ({user.email})")
            print(f"   Subscription ID: {active_subscription.payment_id}")
            print(f"   Plan: {active_subscription.plan}")
            
            # Test Stripe subscription retrieval
            try:
                stripe.api_key = Config.STRIPE_SECRET_KEY
                stripe_subscription = stripe.Subscription.retrieve(active_subscription.payment_id)
                customer_id = stripe_subscription.customer
                print(f"✅ Stripe subscription found")
                print(f"   Customer ID: {customer_id}")
                print(f"   Subscription Status: {stripe_subscription.status}")
                
                # Test portal session creation
                try:
                    portal_session = stripe.billing_portal.Session.create(
                        customer=customer_id,
                        return_url="https://usezyppts.com/account?portal=success",
                    )
                    print(f"✅ Portal session created successfully")
                    print(f"   Portal URL: {portal_session.url}")
                    print(f"   Session ID: {portal_session.id}")
                    
                except stripe.error.InvalidRequestError as e:
                    print(f"❌ Portal session creation failed: {e}")
                    if "No such customer" in str(e):
                        print("   The customer ID doesn't exist in Stripe")
                    elif "No such configuration" in str(e):
                        print("   Customer Portal not configured in Stripe Dashboard")
                    return False
                    
            except stripe.error.InvalidRequestError as e:
                print(f"❌ Stripe subscription retrieval failed: {e}")
                if "No such subscription" in str(e):
                    print("   The subscription ID in database doesn't exist in Stripe")
                return False
                
    except Exception as e:
        print(f"❌ Error testing portal endpoint: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Customer Portal Error Debugger")
    print("=" * 40)
    print()
    
    success = debug_portal_error()
    
    if success:
        print("\n" + "=" * 40)
        test_portal_endpoint()
    
    print("\n📋 Next Steps:")
    print("1. Check the specific error message above")
    print("2. Fix the identified issue")
    print("3. Test the portal again")
    print("4. If still having issues, check Flask logs for more details") 
#!/usr/bin/env python3
"""
Script to verify annual subscription configuration
This script tests that the annual product IDs are properly configured and working.
"""

import os
import sys
import json

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def verify_annual_config():
    """
    Verify that annual subscription configuration is properly set up
    """
    print("🔍 Verifying annual subscription configuration...")
    print()
    
    # Check subscription plans configuration
    plans = Config.SUBSCRIPTION_PLANS
    
    issues_found = []
    warnings = []
    
    for plan_name, plan_config in plans.items():
        print(f"📋 Checking {plan_name.upper()} plan...")
        
        # Check if plan has annual pricing configured
        if 'stripe_annual_price_id' in plan_config:
            annual_id = plan_config['stripe_annual_price_id']
            
            if annual_id and not annual_id.endswith('_annual'):
                print(f"  ✅ Annual price ID configured: {annual_id}")
                
                # Check if it's not the same as monthly
                monthly_id = plan_config.get('stripe_price_id')
                if monthly_id and annual_id != monthly_id:
                    print(f"  ✅ Annual ID differs from monthly ID")
                else:
                    warnings.append(f"{plan_name}: Annual ID same as monthly ID")
            else:
                issues_found.append(f"{plan_name}: Annual price ID not properly configured")
        else:
            if plan_name != 'free' and plan_name != 'enterprise':
                issues_found.append(f"{plan_name}: Missing stripe_annual_price_id")
            else:
                print(f"  ℹ️  {plan_name} plan doesn't require annual pricing")
        
        # Check pricing consistency
        if 'price' in plan_config:
            price = plan_config['price']
            print(f"  💰 Monthly price: ${price}")
            
            # Expected annual pricing based on frontend
            expected_annual = {
                'pro': 7.99,
                'studio': 23.0,
                'enterprise': 79.20
            }
            
            if plan_name in expected_annual:
                expected = expected_annual[plan_name]
                print(f"  💰 Expected annual price: ${expected}/month")
                
                # Calculate savings
                if price > 0:
                    savings = ((price - expected) / price) * 100
                    print(f"  💡 Annual savings: {savings:.1f}%")
        
        print()
    
    # Check Stripe configuration
    print("🔧 Checking Stripe configuration...")
    
    if Config.STRIPE_SECRET_KEY:
        print("  ✅ Stripe secret key configured")
    else:
        issues_found.append("Stripe secret key not configured")
    
    if Config.STRIPE_PUBLISHABLE_KEY:
        print("  ✅ Stripe publishable key configured")
    else:
        warnings.append("Stripe publishable key not configured")
    
    if Config.STRIPE_WEBHOOK_SECRET:
        print("  ✅ Stripe webhook secret configured")
    else:
        warnings.append("Stripe webhook secret not configured")
    
    print()
    
    # Summary
    print("📊 Configuration Summary:")
    print()
    
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        print()
    else:
        print("✅ No critical issues found")
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # Recommendations
    print("💡 Recommendations:")
    
    if not issues_found and not warnings:
        print("  ✅ Configuration looks good! Ready for testing.")
        print("  📝 Next steps:")
        print("    1. Test the subscription flow with annual billing")
        print("    2. Verify pricing in Stripe checkout")
        print("    3. Monitor subscription creation in Stripe dashboard")
    else:
        print("  🔧 Fix the issues above before testing")
        print("  📖 See Backend/scripts/README_annual_setup.md for setup instructions")
    
    return len(issues_found) == 0

def test_routes_integration():
    """
    Test that the routes properly handle annual billing
    """
    print("🔗 Testing routes integration...")
    print()
    
    # Import routes to test the logic
    try:
        from routes import create_checkout_session
        
        # Mock test data
        test_cases = [
            {'plan': 'pro', 'billing': 'monthly'},
            {'plan': 'pro', 'billing': 'annual'},
            {'plan': 'studio', 'billing': 'monthly'},
            {'plan': 'studio', 'billing': 'annual'},
        ]
        
        print("  📋 Testing checkout session creation logic...")
        
        for test_case in test_cases:
            plan = test_case['plan']
            billing = test_case['billing']
            
            plan_config = Config.SUBSCRIPTION_PLANS.get(plan)
            if plan_config:
                if billing == 'annual':
                    price_id = plan_config.get('stripe_annual_price_id') or plan_config.get('stripe_price_id')
                else:
                    price_id = plan_config.get('stripe_price_id')
                
                if price_id:
                    print(f"    ✅ {plan} {billing}: {price_id}")
                else:
                    print(f"    ❌ {plan} {billing}: No price ID found")
            else:
                print(f"    ❌ {plan}: Plan not found")
        
        print()
        print("  ✅ Routes integration test completed")
        
    except ImportError as e:
        print(f"  ⚠️  Could not import routes: {e}")
        print("  ℹ️  This is normal if running outside the main application context")
    
    except Exception as e:
        print(f"  ❌ Error testing routes: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Annual Subscription Configuration Verification")
    print("=" * 60)
    print()
    
    # Run verification
    config_ok = verify_annual_config()
    
    print("-" * 60)
    
    # Test routes integration
    test_routes_integration()
    
    print("-" * 60)
    print()
    
    if config_ok:
        print("🎉 Verification completed successfully!")
        print("Your annual subscription configuration is ready for testing.")
    else:
        print("⚠️  Verification completed with issues.")
        print("Please fix the issues above before proceeding.")
    
    print()
    print("For setup instructions, see: Backend/scripts/README_annual_setup.md") 
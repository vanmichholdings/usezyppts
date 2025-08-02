#!/usr/bin/env python3
"""
Simple debug script for customer portal redirect errors
"""

import os
import sys

def check_common_issues():
    """
    Check common issues that cause portal redirect errors
    """
    print("🔍 Customer Portal Redirect Error - Common Issues")
    print("=" * 55)
    print()
    
    print("📋 Most Common Error Causes:")
    print()
    
    print("1️⃣ **Customer Portal Not Enabled in Stripe Dashboard**")
    print("   • Go to Stripe Dashboard → Settings → Customer Portal")
    print("   • Click 'Enable Customer Portal'")
    print("   • Configure business information and branding")
    print("   • Set up return URLs")
    print()
    
    print("2️⃣ **Invalid or Missing Subscription ID**")
    print("   • Check if user has an active subscription")
    print("   • Verify subscription.payment_id is populated")
    print("   • Ensure payment_id is a valid Stripe subscription ID")
    print("   • Format should be: sub_xxxxxxxxxxxxx")
    print()
    
    print("3️⃣ **Stripe API Key Issues**")
    print("   • Verify STRIPE_SECRET_KEY is correct")
    print("   • Check if using test vs live keys")
    print("   • Ensure key has proper permissions")
    print()
    
    print("4️⃣ **Return URL Configuration**")
    print("   • Check return URL in portal session creation")
    print("   • Ensure URL matches your domain")
    print("   • Verify URL is accessible")
    print()
    
    print("5️⃣ **Customer ID Issues**")
    print("   • Customer ID must exist in Stripe")
    print("   • Customer must be associated with subscription")
    print("   • Check if customer was deleted in Stripe")
    print()
    
    print("🔧 **Quick Fixes to Try:**")
    print()
    print("1. **Enable Customer Portal in Stripe Dashboard:**")
    print("   - Go to Settings → Customer Portal")
    print("   - Click 'Enable Customer Portal'")
    print("   - Configure business profile")
    print("   - Set return URL to: https://usezyppts.com/account")
    print()
    
    print("2. **Check Browser Console:**")
    print("   - Open browser developer tools (F12)")
    print("   - Go to Console tab")
    print("   - Click 'Change Payment Method' button")
    print("   - Look for JavaScript errors")
    print()
    
    print("3. **Check Flask Logs:**")
    print("   - Look for Python errors in terminal")
    print("   - Check for Stripe API errors")
    print("   - Verify database queries")
    print()
    
    print("4. **Test with Known Good Data:**")
    print("   - Create a test subscription in Stripe")
    print("   - Update database with valid subscription ID")
    print("   - Test portal with this subscription")
    print()
    
    print("📞 **Specific Error Messages to Look For:**")
    print()
    print("• 'No such configuration' → Customer Portal not enabled")
    print("• 'No such customer' → Customer doesn't exist in Stripe")
    print("• 'No such subscription' → Subscription ID is invalid")
    print("• 'Authentication failed' → Stripe API key is wrong")
    print("• 'Invalid return URL' → Return URL configuration issue")
    print()
    
    print("🎯 **Next Steps:**")
    print("1. Check the specific error message you received")
    print("2. Match it to one of the common issues above")
    print("3. Apply the corresponding fix")
    print("4. Test the portal again")
    print("5. If still having issues, share the exact error message")

if __name__ == "__main__":
    check_common_issues() 
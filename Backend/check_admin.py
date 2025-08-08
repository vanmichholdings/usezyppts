#!/usr/bin/env python3
"""
Script to check Mike's admin account status
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_admin():
    """Check Mike's admin account status"""
    
    try:
        # Import Flask app and models
        from app_config import create_app
        from models import User, Subscription
        
        # Create app context
        app = create_app()
        
        with app.app_context():
            # Check Mike's admin account
            mike_email = "mike@usezyppts.com"
            mike_user = User.query.filter_by(email=mike_email).first()
            
            if mike_user:
                print("\n✅ Mike's admin account exists:")
                print(f"   Email: {mike_email}")
                print(f"   Username: {mike_user.username}")
                print(f"   Admin status: {mike_user.is_admin}")
                print(f"   Active status: {mike_user.is_active}")
                
                # Check subscription
                subscription = Subscription.query.filter_by(user_id=mike_user.id).first()
                if subscription:
                    print(f"\n✅ Subscription found:")
                    print(f"   Plan: {subscription.plan}")
                    print(f"   Status: {subscription.status}")
                    print(f"   Credits: {subscription.monthly_credits}")
                else:
                    print("\n❌ No subscription found")
            else:
                print("\n❌ Mike's admin account not found")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Error checking admin account: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Checking Mike's Admin Account Status")
    print("=" * 40)
    check_admin() 
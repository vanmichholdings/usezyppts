#!/usr/bin/env python3
"""
Script to test login functionality
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_login():
    """Test login functionality"""

    try:
        # Import Flask app and models
        from app_config import create_app
        from models import User
        from werkzeug.security import check_password_hash

        # Create app context
        app = create_app()

        with app.app_context():
            # Check Mike's admin account
            mike_email = "mike@usezyppts.com"
            mike_user = User.query.filter_by(email=mike_email).first()

            if mike_user:
                print("\n✅ Mike's admin account found:")
                print(f"   Email: {mike_email}")
                print(f"   Username: {mike_user.username}")
                print(f"   Admin status: {mike_user.is_admin}")
                print(f"   Active status: {mike_user.is_active}")
                print(f"   Password hash: {mike_user.password_hash[:20]}...")

                # Test password verification
                test_password = "mike123"
                if mike_user.check_password(test_password):
                    print(f"✅ Password verification successful for '{test_password}'")
                else:
                    print(f"❌ Password verification failed for '{test_password}'")
                    
                    # Try to set a new password
                    print("🔄 Setting new password...")
                    mike_user.set_password(test_password)
                    from app_config import db
                    db.session.commit()
                    print("✅ New password set successfully")
                    
                    # Test again
                    if mike_user.check_password(test_password):
                        print(f"✅ Password verification successful after reset")
                    else:
                        print(f"❌ Password verification still failed after reset")

            else:
                print("\n❌ Mike's admin account not found")

            return True

    except Exception as e:
        print(f"\n❌ Error testing login: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Login Functionality")
    print("=" * 40)
    test_login() 
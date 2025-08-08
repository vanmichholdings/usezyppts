#!/usr/bin/env python3
"""
Script to create Mike's admin user using Flask app context
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_admin_flask():
    """Create Mike's admin user using Flask app context"""
    
    try:
        # Import Flask app and models
        from app_config import create_app, db
        from models import User, Subscription
        
        # Create app context
        app = create_app()
        
        with app.app_context():
            print("🔧 Creating Mike's Admin User Account (Flask Context)")
            print("=" * 50)
            
            # Check if Mike's admin user already exists
            mike_email = "mike@usezyppts.com"
            existing_mike = User.query.filter_by(email=mike_email).first()
            
            if existing_mike:
                print(f"✅ Mike's admin user already exists:")
                print(f"   ID: {existing_mike.id}")
                print(f"   Username: {existing_mike.username}")
                print(f"   Email: {existing_mike.email}")
                print(f"   Admin: {existing_mike.is_admin}")
                print(f"   Active: {existing_mike.is_active}")
                
                # Ensure admin privileges are set
                if not existing_mike.is_admin:
                    existing_mike.is_admin = True
                    db.session.commit()
                    print("✅ Admin privileges granted to existing user")
                
                # Check subscription
                if existing_mike.subscription:
                    print(f"   Subscription: {existing_mike.subscription.plan} ({existing_mike.subscription.status})")
                else:
                    # Create subscription for existing user
                    mike_subscription = Subscription(
                        user_id=existing_mike.id,
                        plan='enterprise',
                        status='active',
                        monthly_credits=999999,
                        used_credits=0,
                        start_date=datetime.utcnow(),
                        auto_renew=True
                    )
                    db.session.add(mike_subscription)
                    db.session.commit()
                    print("✅ Enterprise subscription created for existing user")
                
                return True
            
            # Create new Mike admin user
            mike_user = User(
                username="mike",
                email=mike_email,
                is_admin=True,
                is_active=True,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            mike_user.set_password("mike123")  # Change this in production!
            
            # Add to database
            db.session.add(mike_user)
            db.session.commit()
            
            print("✅ Mike's admin user created successfully!")
            print(f"   ID: {mike_user.id}")
            print(f"   Email: {mike_email}")
            print(f"   Username: mike")
            print(f"   Password: mike123")
            print(f"   Admin: {mike_user.is_admin}")
            print(f"   Active: {mike_user.is_active}")
            print("\n⚠️  IMPORTANT: Change the password in production!")
            
            # Create a subscription for Mike
            mike_subscription = Subscription(
                user_id=mike_user.id,
                plan='enterprise',
                status='active',
                monthly_credits=999999,
                used_credits=0,
                start_date=datetime.utcnow(),
                auto_renew=True
            )
            
            db.session.add(mike_subscription)
            db.session.commit()
            
            print("✅ Enterprise subscription created for Mike")
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating Mike's admin user: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_admin_flask()
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Log in with mike@usezyppts.com / mike123")
        print("   2. Visit: https://zyppts-logo-processor-aged-violet-9912.fly.dev/admin/")
        print("   3. Change the password in the admin panel")
    else:
        print("\n❌ Failed to create admin user") 
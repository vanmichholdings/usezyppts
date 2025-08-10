#!/usr/bin/env python3
"""
Script to create an admin user for testing
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_admin_user():
    """Create an admin user for testing"""
    
    try:
        # Import Flask app and models
        from app_config import create_app
        from models import db, User, Subscription
        
        # Create app context
        app = create_app()
        
        with app.app_context():
            # Check if admin user already exists
            admin_email = "mike@usezyppts.com"
            existing_admin = User.query.filter_by(email=admin_email).first()
            
            if existing_admin:
                print(f"✅ Admin user already exists: {admin_email}")
                print(f"   Username: {existing_admin.username}")
                print(f"   Admin status: {existing_admin.is_admin}")
                return existing_admin
            
            # Create new admin user
            admin_user = User(
                username="mike",
                email=admin_email,
                is_admin=True,
                is_active=True,
                created_at=datetime.utcnow()
            )
            admin_user.set_password("admin123")  # Change this in production!
            
            # Add to database
            db.session.add(admin_user)
            db.session.commit()
            
            print("✅ Admin user created successfully!")
            print(f"   Email: {admin_email}")
            print(f"   Username: mike")
            print(f"   Password: admin123")
            print("\n⚠️  IMPORTANT: Change the password in production!")
            
            # Create a subscription for the admin user
            admin_subscription = Subscription(
                user_id=admin_user.id,
                plan='enterprise',
                status='active',
                monthly_credits=999999,
                used_credits=0,
                start_date=datetime.utcnow(),
                auto_renew=True
            )
            
            db.session.add(admin_subscription)
            db.session.commit()
            
            print("✅ Admin subscription created (Enterprise plan)")
            
            return admin_user
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return None

if __name__ == "__main__":
    print("🔧 Creating Admin User for Testing")
    print("=" * 40)
    create_admin_user()
    print("\n🎯 Next Steps:")
    print("   1. Start the Flask application: python run.py")
    print("   2. Log in with mike@usezyppts.com / admin123")
    print("   3. Visit: http://localhost:5003/admin/analytics") 
#!/usr/bin/env python3
"""
Database initialization script for Zyppts logo processor.
This script ensures the database is properly set up with required tables and admin user.
"""

import os
import sys
from datetime import datetime

# Add the Backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app_config import create_app, db
    from models import User, Subscription
    from config import Config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def init_database():
    """Initialize the database with tables and admin user"""
    app = create_app()
    
    with app.app_context():
        try:
            print("Initializing database...")
            
            # Create all tables
            db.create_all()
            print("✓ Database tables created")
            
            # Check if mike user exists (your original user)
            mike_user = User.query.filter_by(email='mike@usezyppts.com').first()
            if not mike_user:
                # Create mike user
                mike_user = User(
                    username='mike',
                    email='mike@usezyppts.com',
                    is_admin=True,
                    is_active=True,
                    is_beta=True,
                    created_at=datetime.utcnow()
                )
                mike_user.set_password('zyppts2024!')  # You should change this
                
                # Create mike subscription
                mike_subscription = Subscription(
                    user=mike_user,
                    plan='enterprise',
                    status='active',
                    monthly_credits=-1,  # Unlimited
                    start_date=datetime.utcnow(),
                    billing_cycle='annual'
                )
                
                db.session.add(mike_user)
                db.session.add(mike_subscription)
                db.session.commit()
                print("✓ Mike user created (username: mike, email: mike@usezyppts.com)")
            else:
                print("✓ Mike user already exists")
                # Update mike user to ensure proper settings
                mike_user.is_admin = True
                mike_user.is_active = True
                mike_user.is_beta = True
                if not mike_user.subscription:
                    mike_subscription = Subscription(
                        user=mike_user,
                        plan='enterprise',
                        status='active',
                        monthly_credits=-1,  # Unlimited
                        start_date=datetime.utcnow(),
                        billing_cycle='annual'
                    )
                    db.session.add(mike_subscription)
                db.session.commit()
                print("✓ Mike user settings updated")
            
            print("\n✅ Database initialization completed successfully!")
            print("\nMike login credentials:")
            print("  Username: mike")
            print("  Email: mike@usezyppts.com")
            print("  Password: zyppts2024!")
            print("\n⚠️  Please change these passwords after first login!")
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            db.session.rollback()
            return False
        
        return True

if __name__ == '__main__':
    if init_database():
        print("\n🚀 Ready to deploy!")
        sys.exit(0)
    else:
        print("\n💥 Initialization failed!")
        sys.exit(1) 
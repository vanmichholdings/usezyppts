#!/usr/bin/env python3
"""
Direct database script to create Mike's admin user
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash

def create_admin_direct():
    """Create Mike's admin user directly in database"""
    
    try:
        # Get database URL from environment
        database_url = os.environ.get('DATABASE_URL')
        if database_url and database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        if not database_url:
            print("❌ DATABASE_URL not found in environment")
            return False
        
        # Create database engine
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        # Check if user already exists
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, username, email, is_admin FROM users WHERE email = :email"), 
                                {"email": "mike@usezyppts.com"})
            existing_user = result.fetchone()
            
            if existing_user:
                print(f"✅ Mike's admin user already exists:")
                print(f"   ID: {existing_user[0]}")
                print(f"   Username: {existing_user[1]}")
                print(f"   Email: {existing_user[2]}")
                print(f"   Admin: {existing_user[3]}")
                
                # Ensure admin privileges
                if not existing_user[3]:
                    conn.execute(text("UPDATE users SET is_admin = true WHERE email = :email"), 
                               {"email": "mike@usezyppts.com"})
                    conn.commit()
                    print("✅ Admin privileges granted to existing user")
                
                # Check subscription
                result = conn.execute(text("SELECT plan, status FROM subscriptions WHERE user_id = :user_id"), 
                                    {"user_id": existing_user[0]})
                subscription = result.fetchone()
                
                if subscription:
                    print(f"   Subscription: {subscription[0]} ({subscription[1]})")
                else:
                    # Create subscription
                    conn.execute(text("""
                        INSERT INTO subscriptions (user_id, plan, status, monthly_credits, used_credits, start_date, auto_renew)
                        VALUES (:user_id, 'enterprise', 'active', 999999, 0, :start_date, true)
                    """), {
                        "user_id": existing_user[0],
                        "start_date": datetime.utcnow()
                    })
                    conn.commit()
                    print("✅ Enterprise subscription created for existing user")
                
                return True
            
            # Create new user
            password_hash = generate_password_hash("mike123")
            now = datetime.utcnow()
            
            # Insert new user
            result = conn.execute(text("""
                INSERT INTO users (username, email, password_hash, is_admin, is_active, created_at, last_login)
                VALUES (:username, :email, :password_hash, :is_admin, :is_active, :created_at, :last_login)
                RETURNING id
            """), {
                "username": "mike",
                "email": "mike@usezyppts.com",
                "password_hash": password_hash,
                "is_admin": True,
                "is_active": True,
                "created_at": now,
                "last_login": now
            })
            
            user_id = result.fetchone()[0]
            conn.commit()
            
            print("✅ Mike's admin user created successfully!")
            print(f"   ID: {user_id}")
            print(f"   Email: mike@usezyppts.com")
            print(f"   Username: mike")
            print(f"   Password: mike123")
            print(f"   Admin: True")
            
            # Create subscription
            conn.execute(text("""
                INSERT INTO subscriptions (user_id, plan, status, monthly_credits, used_credits, start_date, auto_renew)
                VALUES (:user_id, 'enterprise', 'active', 999999, 0, :start_date, true)
            """), {
                "user_id": user_id,
                "start_date": now
            })
            conn.commit()
            
            print("✅ Enterprise subscription created for Mike")
            print("\n⚠️  IMPORTANT: Change the password in production!")
            
            return True
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Creating Mike's Admin User Account (Direct Database)")
    print("=" * 50)
    success = create_admin_direct()
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Log in with mike@usezyppts.com / mike123")
        print("   2. Visit: https://zyppts-logo-processor-aged-violet-9912.fly.dev/admin/")
        print("   3. Change the password in the admin panel")
    else:
        print("\n❌ Failed to create admin user") 
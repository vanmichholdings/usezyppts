#!/usr/bin/env python3
"""
Migrate from SQLite to PostgreSQL for Render deployment
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from app_config import create_app, db
from models import User, Subscription, LogoUpload, LogoVariation

def migrate_to_postgres():
    """Migrate data to PostgreSQL"""
    app = create_app()
    
    with app.app_context():
        print("🔄 Creating database tables...")
        
        # Create all tables
        db.create_all()
        
        # Create indexes for better performance
        print("📊 Creating database indexes...")
        
        # User table indexes
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_user_email ON users (email);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_user_created_at ON users (created_at);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_user_is_admin ON users (is_admin);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_user_is_active ON users (is_active);')
        
        # Subscription table indexes
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_subscription_user_id ON subscriptions (user_id);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_subscription_status ON subscriptions (status);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_subscription_plan ON subscriptions (plan);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_subscription_start_date ON subscriptions (start_date);')
        
        # Logo upload indexes
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_upload_user_id ON logo_uploads (user_id);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_upload_status ON logo_uploads (status);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_upload_date ON logo_uploads (upload_date);')
        
        # Logo variation indexes
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_variation_upload_id ON logo_variations (upload_id);')
        db.session.execute('CREATE INDEX IF NOT EXISTS idx_variation_type ON logo_variations (variation_type);')
        
        db.session.commit()
        
        print("✅ Database migration complete!")
        print("\n📋 Next Steps:")
        print("1. Your admin system is now ready for live tracking")
        print("2. Access admin panel at: https://your-app.onrender.com/admin")
        print("3. Monitor logs in Render dashboard")
        print("4. Set up admin user with: python scripts/add_is_admin_column.py")

def create_admin_user():
    """Create admin user for production"""
    app = create_app()
    
    with app.app_context():
        # Check if admin user exists
        admin_user = User.query.filter_by(email='test@zyppts.com').first()
        
        if admin_user:
            admin_user.is_admin = True
            db.session.commit()
            print("✅ Admin user updated: test@zyppts.com")
        else:
            print("❌ Admin user not found. Please create user first.")
            print("   You can create a user through the registration page")

def check_database_connection():
    """Check database connection and health"""
    app = create_app()
    
    with app.app_context():
        try:
            # Test database connection
            db.session.execute('SELECT 1')
            print("✅ Database connection successful")
            
            # Check table counts
            user_count = User.query.count()
            subscription_count = Subscription.query.count()
            upload_count = LogoUpload.query.count()
            
            print(f"📊 Database statistics:")
            print(f"   Users: {user_count}")
            print(f"   Subscriptions: {subscription_count}")
            print(f"   Uploads: {upload_count}")
            
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_database_connection()
        elif sys.argv[1] == "admin":
            create_admin_user()
        else:
            print("Usage: python migrate_to_postgres.py [check|admin]")
    else:
        migrate_to_postgres() 
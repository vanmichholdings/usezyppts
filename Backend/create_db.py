#!/usr/bin/env python3
"""
Create database with promo code support
"""

import os
import sys

def create_database():
    """Create the database with all tables"""
    print("🗄️ Creating database with promo code support...")
    
    try:
        from app_config import create_app, db
        from models import User, Subscription, LogoUpload, LogoVariation
        
        app = create_app()
        
        with app.app_context():
            # Drop all tables and recreate them
            db.drop_all()
            print("✅ Dropped existing tables")
            
            # Create all tables
            db.create_all()
            print("✅ Created all tables")
            
            # Verify the users table structure
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            columns = inspector.get_columns('users')
            
            print("\n📋 Users table structure:")
            for column in columns:
                print(f"   - {column['name']}: {column['type']}")
            
            # Check if promo_code and upload_credits columns exist
            column_names = [col['name'] for col in columns]
            if 'promo_code' in column_names and 'upload_credits' in column_names:
                print("\n✅ Promo code columns successfully added!")
                print("   - promo_code: VARCHAR(50)")
                print("   - upload_credits: INTEGER DEFAULT 3")
            else:
                print("\n❌ Promo code columns missing!")
                print(f"   Available columns: {column_names}")
            
            print("\n🎉 Database creation completed successfully!")
            
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_database() 
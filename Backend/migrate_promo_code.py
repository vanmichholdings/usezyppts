#!/usr/bin/env python3
"""
Migration script to add promo code support to existing database
"""

import sqlite3
import os

def migrate_database():
    """Add promo code and upload credits columns to existing database"""
    print("🔄 Migrating database for promo code support...")
    
    # Database file path
    db_path = 'app.db'
    
    if not os.path.exists(db_path):
        print("❌ Database file not found. Creating new database...")
        from app_config import create_app, db
        app = create_app()
        with app.app_context():
            db.create_all()
        print("✅ New database created with promo code support")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        print(f"Current columns in users table: {columns}")
        
        # Add promo_code column if it doesn't exist
        if 'promo_code' not in columns:
            print("Adding promo_code column...")
            cursor.execute("ALTER TABLE users ADD COLUMN promo_code VARCHAR(50)")
            print("✅ Added promo_code column")
        else:
            print("✅ promo_code column already exists")
        
        # Add upload_credits column if it doesn't exist
        if 'upload_credits' not in columns:
            print("Adding upload_credits column...")
            cursor.execute("ALTER TABLE users ADD COLUMN upload_credits INTEGER DEFAULT 3")
            print("✅ Added upload_credits column")
        else:
            print("✅ upload_credits column already exists")
        
        # Update existing users to have default upload_credits
        cursor.execute("UPDATE users SET upload_credits = 3 WHERE upload_credits IS NULL")
        updated_rows = cursor.rowcount
        print(f"✅ Updated {updated_rows} existing users with default upload_credits")
        
        # Commit changes
        conn.commit()
        print("✅ Database migration completed successfully")
        
        # Verify the migration
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"Updated columns in users table: {columns}")
        
        # Check if promo_code and upload_credits are present
        if 'promo_code' in columns and 'upload_credits' in columns:
            print("✅ Migration verification successful")
        else:
            print("❌ Migration verification failed")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database() 
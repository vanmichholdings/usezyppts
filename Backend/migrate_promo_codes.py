#!/usr/bin/env python3
"""
Migration script to add promo code columns to existing database
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def migrate_promo_codes():
    """Add promo code columns to existing database"""
    
    try:
        # Import Flask app and models
        from app_config import create_app, db
        from models import User, PromoCode
        
        # Create app context
        app = create_app()
        
        with app.app_context():
            print("🚀 Starting promo code migration...")
            
            # Check if promo code columns already exist
            try:
                # Try to query the promo code columns
                result = db.session.execute(db.text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result.fetchall()]
                
                if 'promo_code_used' in columns and 'promo_code_applied' in columns:
                    print("✅ Promo code columns already exist in users table")
                else:
                    print("📝 Adding promo code columns to users table...")
                    
                    # Add promo code columns
                    db.session.execute(db.text("""
                        ALTER TABLE users 
                        ADD COLUMN promo_code_used VARCHAR(50)
                    """))
                    
                    db.session.execute(db.text("""
                        ALTER TABLE users 
                        ADD COLUMN promo_code_applied BOOLEAN DEFAULT FALSE
                    """))
                    
                    db.session.commit()
                    print("✅ Promo code columns added successfully")
                    
            except Exception as e:
                print(f"⚠️ Error checking/adding columns: {e}")
                # Try alternative approach for PostgreSQL
                try:
                    db.session.execute(db.text("""
                        ALTER TABLE users 
                        ADD COLUMN IF NOT EXISTS promo_code_used VARCHAR(50)
                    """))
                    
                    db.session.execute(db.text("""
                        ALTER TABLE users 
                        ADD COLUMN IF NOT EXISTS promo_code_applied BOOLEAN DEFAULT FALSE
                    """))
                    
                    db.session.commit()
                    print("✅ Promo code columns added successfully (PostgreSQL)")
                except Exception as e2:
                    print(f"❌ Failed to add columns: {e2}")
                    return False
            
            # Create promo_codes table if it doesn't exist
            try:
                # Check if promo_codes table exists
                result = db.session.execute(db.text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='promo_codes'
                """))
                
                if not result.fetchone():
                    print("📝 Creating promo_codes table...")
                    
                    # Create promo_codes table
                    db.session.execute(db.text("""
                        CREATE TABLE promo_codes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            code VARCHAR(50) UNIQUE NOT NULL,
                            description VARCHAR(200),
                            discount_percentage INTEGER DEFAULT 0,
                            discount_amount FLOAT DEFAULT 0.0,
                            max_uses INTEGER,
                            current_uses INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            expires_at DATETIME,
                            early_access BOOLEAN DEFAULT FALSE,
                            bonus_credits INTEGER DEFAULT 0,
                            plan_upgrade VARCHAR(20)
                        )
                    """))
                    
                    db.session.commit()
                    print("✅ Promo_codes table created successfully")
                else:
                    print("✅ Promo_codes table already exists")
                    
            except Exception as e:
                print(f"⚠️ Error creating promo_codes table: {e}")
                # Try PostgreSQL approach
                try:
                    db.session.execute(db.text("""
                        CREATE TABLE IF NOT EXISTS promo_codes (
                            id SERIAL PRIMARY KEY,
                            code VARCHAR(50) UNIQUE NOT NULL,
                            description VARCHAR(200),
                            discount_percentage INTEGER DEFAULT 0,
                            discount_amount FLOAT DEFAULT 0.0,
                            max_uses INTEGER,
                            current_uses INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP,
                            early_access BOOLEAN DEFAULT FALSE,
                            bonus_credits INTEGER DEFAULT 0,
                            plan_upgrade VARCHAR(20)
                        )
                    """))
                    
                    db.session.commit()
                    print("✅ Promo_codes table created successfully (PostgreSQL)")
                except Exception as e2:
                    print(f"❌ Failed to create promo_codes table: {e2}")
                    return False
            
            # Create EARLYZYPPTS promo code
            try:
                existing_code = db.session.execute(
                    db.text("SELECT id FROM promo_codes WHERE code = 'EARLYZYPPTS'")
                ).fetchone()
                
                if existing_code:
                    print("✅ EARLYZYPPTS promo code already exists")
                else:
                    print("📝 Creating EARLYZYPPTS promo code...")
                    
                    db.session.execute(db.text("""
                        INSERT INTO promo_codes (
                            code, description, early_access, bonus_credits, 
                            plan_upgrade, max_uses, is_active, expires_at,
                            discount_percentage, discount_amount
                        ) VALUES (
                            'EARLYZYPPTS', 
                            'Early Access - Pro features with free credits',
                            TRUE, 0, NULL, NULL, TRUE, NULL, 0, 0.0
                        )
                    """))
                    
                    db.session.commit()
                    print("✅ EARLYZYPPTS promo code created successfully")
                    
            except Exception as e:
                print(f"❌ Error creating EARLYZYPPTS promo code: {e}")
                return False
            
            print("\n🎉 Promo code migration completed successfully!")
            print("\n🎯 EARLYZYPPTS Promo Code Features:")
            print("   - Gives early access to pro/studio features")
            print("   - Users stay on free plan (3 credits)")
            print("   - Access to advanced effects, batch processing, API")
            print("   - After credits run out, users are prompted to upgrade")
            print("   - Unlimited uses")
            print("   - No expiration date")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = migrate_promo_codes()
    if success:
        print("\n🎉 Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1) 
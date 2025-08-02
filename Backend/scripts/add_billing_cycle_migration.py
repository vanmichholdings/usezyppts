#!/usr/bin/env python3
"""
Database Migration: Add billing_cycle field to subscriptions table
"""

import os
import sys
from pathlib import Path

def add_billing_cycle_migration():
    """Add billing_cycle field to subscriptions table"""
    print("🔄 Adding billing_cycle field to subscriptions table...")
    print("=" * 60)
    
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.append(str(backend_dir))
    
    try:
        from app_config import create_app, db
        from models import Subscription
        
        app = create_app()
        
        with app.app_context():
            # Check if the column already exists
            inspector = db.inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('subscriptions')]
            
            if 'billing_cycle' in columns:
                print("✅ billing_cycle column already exists")
                return True
            
            # Add the new column using raw SQL
            print("📝 Adding billing_cycle column...")
            with db.engine.connect() as conn:
                conn.execute(db.text('ALTER TABLE subscriptions ADD COLUMN billing_cycle VARCHAR(20) DEFAULT "monthly"'))
                conn.commit()
            
            # Update existing records to have 'monthly' as default
            print("🔄 Updating existing records...")
            with db.engine.connect() as conn:
                conn.execute(db.text('UPDATE subscriptions SET billing_cycle = "monthly" WHERE billing_cycle IS NULL'))
                conn.commit()
            
            print("✅ Migration completed successfully!")
            print("\n📋 Migration Summary:")
            print("- Added billing_cycle column to subscriptions table")
            print("- Set default value to 'monthly'")
            print("- Updated existing records")
            
            return True
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_migration():
    """Verify the migration was successful"""
    print("\n🔍 Verifying migration...")
    print("=" * 60)
    
    try:
        from app_config import create_app, db
        from models import Subscription
        
        app = create_app()
        
        with app.app_context():
            # Check if the column exists
            inspector = db.inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('subscriptions')]
            
            if 'billing_cycle' in columns:
                print("✅ billing_cycle column exists")
                
                # Check a few records
                subscriptions = Subscription.query.limit(5).all()
                print(f"📊 Found {len(subscriptions)} subscription records")
                
                for sub in subscriptions:
                    print(f"  - User {sub.user_id}: {sub.plan} ({sub.billing_cycle})")
                
                return True
            else:
                print("❌ billing_cycle column not found")
                return False
                
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Database Migration: Add billing_cycle field")
    print("=" * 60)
    
    # Run migration
    if add_billing_cycle_migration():
        print("\n" + "=" * 60)
        
        # Verify migration
        verify_migration()
    
    print("\n📖 Next Steps:")
    print("1. Restart your Flask application")
    print("2. Test the subscription toggle functionality")
    print("3. Verify that billing information is stored correctly")

if __name__ == "__main__":
    main() 
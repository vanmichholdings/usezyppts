#!/usr/bin/env python3
"""
Update free plan users to have 3 monthly credits
This script updates all existing free plan users to have the new credit limit.
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def update_free_plan_credits():
    """
    Update all free plan users to have 3 monthly credits
    """
    print("🔄 Updating Free Plan Credits")
    print("=" * 40)
    print()
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        
        app = create_app()
        with app.app_context():
            
            # Find all free plan subscriptions
            free_subscriptions = Subscription.query.filter_by(plan='free').all()
            
            if not free_subscriptions:
                print("ℹ️  No free plan subscriptions found")
                return True
            
            print(f"📋 Found {len(free_subscriptions)} free plan subscription(s)")
            print()
            
            updated_count = 0
            
            for subscription in free_subscriptions:
                user = subscription.user
                old_credits = subscription.monthly_credits
                
                # Only update if credits are not already 3
                if old_credits != 3:
                    subscription.monthly_credits = 3
                    updated_count += 1
                    
                    print(f"✅ Updated user: {user.username} ({user.email})")
                    print(f"   Old credits: {old_credits} → New credits: 3")
                    print(f"   Used credits: {subscription.used_credits}")
                    print()
                else:
                    print(f"ℹ️  User {user.username} already has 3 credits")
                    print()
            
            if updated_count > 0:
                # Commit the changes
                db.session.commit()
                print(f"🎉 Successfully updated {updated_count} free plan user(s)")
                print()
                
                # Show summary
                print("📊 Summary:")
                print(f"   • Total free plan users: {len(free_subscriptions)}")
                print(f"   • Users updated: {updated_count}")
                print(f"   • Users already correct: {len(free_subscriptions) - updated_count}")
            else:
                print("ℹ️  No updates needed - all free plan users already have 3 credits")
            
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the Backend directory")
        return False
    except Exception as e:
        print(f"❌ Error updating free plan credits: {e}")
        return False

def show_current_free_plan_status():
    """
    Show current status of free plan users
    """
    print("📊 Current Free Plan Status")
    print("=" * 30)
    print()
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        
        app = create_app()
        with app.app_context():
            
            # Find all free plan subscriptions
            free_subscriptions = Subscription.query.filter_by(plan='free').all()
            
            if not free_subscriptions:
                print("ℹ️  No free plan subscriptions found")
                return True
            
            print(f"📋 Found {len(free_subscriptions)} free plan subscription(s):")
            print()
            
            for subscription in free_subscriptions:
                user = subscription.user
                print(f"👤 {user.username} ({user.email})")
                print(f"   Monthly Credits: {subscription.monthly_credits}")
                print(f"   Used Credits: {subscription.used_credits}")
                print(f"   Remaining: {subscription.monthly_credits - subscription.used_credits}")
                print(f"   Status: {subscription.status}")
                print()
            
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error showing status: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update free plan credits to 3 per month")
    parser.add_argument("--status", action="store_true", help="Show current status only")
    parser.add_argument("--update", action="store_true", help="Update free plan credits")
    
    args = parser.parse_args()
    
    if args.status:
        show_current_free_plan_status()
    elif args.update:
        update_free_plan_credits()
    else:
        print("Please specify an option:")
        print("  --status : Show current free plan status")
        print("  --update : Update free plan credits to 3")
        print()
        print("Example: python update_free_plan_credits.py --update") 
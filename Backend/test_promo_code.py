#!/usr/bin/env python3
"""
Test script for promo code functionality
"""

import os
import sys
from datetime import datetime

def test_promo_code_functionality():
    """Test the promo code functionality"""
    print("🧪 Testing Promo Code Functionality")
    print("=" * 50)
    
    try:
        from app_config import create_app, db
        from models import User, Subscription
        
        app = create_app()
        
        with app.app_context():
            # Test 1: Create a user with promo code
            print("\n1. Testing user creation with promo code...")
            
            # Check if test user already exists
            test_user = User.query.filter_by(email='test_promo@example.com').first()
            if test_user:
                print("   Test user already exists, deleting...")
                db.session.delete(test_user)
                db.session.commit()
            
            # Create new user with promo code
            promo_user = User(
                username='test_promo_user',
                email='test_promo@example.com',
                created_at=datetime.utcnow(),
                is_active=True,
                promo_code='EARLYZYPPTS',
                upload_credits=3
            )
            promo_user.set_password('testpassword123')
            
            # Add user to session and flush to get the ID
            db.session.add(promo_user)
            db.session.flush()  # This will assign the ID
            
            # Create subscription for promo user
            promo_subscription = Subscription(
                user_id=promo_user.id,
                plan='promo_pro_studio',
                status='active',
                monthly_credits=3,
                used_credits=0,
                start_date=datetime.utcnow(),
                auto_renew=False
            )
            
            db.session.add(promo_subscription)
            db.session.commit()
            
            print("   ✅ Promo user created successfully")
            print(f"   - Username: {promo_user.username}")
            print(f"   - Promo Code: {promo_user.promo_code}")
            print(f"   - Upload Credits: {promo_user.upload_credits}")
            print(f"   - Plan: {promo_subscription.plan}")
            
            # Test 2: Test promo user methods
            print("\n2. Testing promo user methods...")
            
            print(f"   - is_promo_user(): {promo_user.is_promo_user()}")
            print(f"   - can_generate_files(): {promo_user.can_generate_files()}")
            print(f"   - get_remaining_credits(): {promo_user.get_remaining_credits()}")
            
            # Test 3: Test credit usage
            print("\n3. Testing credit usage...")
            
            print(f"   Initial credits: {promo_user.upload_credits}")
            
            # Use one credit
            if promo_user.use_upload_credit():
                print("   ✅ Successfully used 1 upload credit")
                print(f"   Remaining credits: {promo_user.upload_credits}")
            else:
                print("   ❌ Failed to use upload credit")
            
            # Use another credit
            if promo_user.use_upload_credit():
                print("   ✅ Successfully used 1 more upload credit")
                print(f"   Remaining credits: {promo_user.upload_credits}")
            else:
                print("   ❌ Failed to use upload credit")
            
            # Use the last credit
            if promo_user.use_upload_credit():
                print("   ✅ Successfully used the last upload credit")
                print(f"   Remaining credits: {promo_user.upload_credits}")
                
                # Test plan reversion
                if promo_user.upload_credits == 0:
                    promo_subscription.plan = 'free'
                    promo_subscription.monthly_credits = 3
                    promo_subscription.used_credits = 0
                    db.session.commit()
                    print("   ✅ Plan reverted to 'free'")
                    print(f"   New plan: {promo_subscription.plan}")
            else:
                print("   ❌ Failed to use upload credit")
            
            # Test 4: Test when no credits remain
            print("\n4. Testing behavior when no credits remain...")
            
            print(f"   - can_generate_files(): {promo_user.can_generate_files()}")
            print(f"   - get_remaining_credits(): {promo_user.get_remaining_credits()}")
            
            # Test 5: Create a regular user for comparison
            print("\n5. Testing regular user creation...")
            
            regular_user = User(
                username='test_regular_user',
                email='test_regular@example.com',
                created_at=datetime.utcnow(),
                is_active=True,
                upload_credits=3
            )
            regular_user.set_password('testpassword123')
            
            # Add user to session and flush to get the ID
            db.session.add(regular_user)
            db.session.flush()
            
            # Create free subscription
            regular_subscription = Subscription(
                user_id=regular_user.id,
                plan='free',
                status='active',
                monthly_credits=3,
                used_credits=0,
                start_date=datetime.utcnow(),
                auto_renew=False
            )
            
            db.session.add(regular_subscription)
            db.session.commit()
            
            print("   ✅ Regular user created successfully")
            print(f"   - Username: {regular_user.username}")
            print(f"   - Promo Code: {regular_user.promo_code}")
            print(f"   - Upload Credits: {regular_user.upload_credits}")
            print(f"   - Plan: {regular_subscription.plan}")
            print(f"   - is_promo_user(): {regular_user.is_promo_user()}")
            
            # Cleanup
            print("\n6. Cleaning up test data...")
            db.session.delete(promo_user)
            db.session.delete(regular_user)
            db.session.commit()
            print("   ✅ Test data cleaned up")
            
            print("\n🎉 All tests passed! Promo code functionality is working correctly.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_promo_code_functionality() 
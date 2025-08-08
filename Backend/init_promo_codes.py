#!/usr/bin/env python3
"""
Script to initialize promo codes in the database
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def init_promo_codes():
    """Initialize promo codes in the database"""
    
    try:
        # Import Flask app and models
        from app_config import create_app, db
        from models import PromoCode
        
        # Create app context
        app = create_app()
        
        with app.app_context():
            print("🚀 Initializing promo codes...")
            
            # Create EARLYZYPPTS promo code
            existing_code = PromoCode.query.filter_by(code='EARLYZYPPTS').first()
            
            if existing_code:
                print(f"✅ EARLYZYPPTS promo code already exists:")
                print(f"   Code: {existing_code.code}")
                print(f"   Description: {existing_code.description}")
                print(f"   Early Access: {existing_code.early_access}")
                print(f"   Active: {existing_code.is_active}")
                print(f"   Uses: {existing_code.current_uses}")
            else:
                # Create the EARLYZYPPTS promo code
                promo_code = PromoCode(
                    code='EARLYZYPPTS',
                    description='Early Access - Pro features with free credits',
                    early_access=True,
                    bonus_credits=0,
                    plan_upgrade=None,
                    max_uses=None,
                    is_active=True,
                    expires_at=None,
                    discount_percentage=0,
                    discount_amount=0.0
                )
                
                db.session.add(promo_code)
                db.session.commit()
                
                print("✅ EARLYZYPPTS promo code created successfully!")
                print(f"   Code: {promo_code.code}")
                print(f"   Description: {promo_code.description}")
                print(f"   Early Access: {promo_code.early_access}")
                print(f"   Active: {promo_code.is_active}")
            
            print("\n🎯 Promo Code Features:")
            print("   - Gives early access to pro/studio features")
            print("   - Users stay on free plan (3 credits)")
            print("   - Access to advanced effects, batch processing, API")
            print("   - After credits run out, users are prompted to upgrade")
            print("   - Unlimited uses")
            print("   - No expiration date")
            
            return True
            
    except Exception as e:
        print(f"❌ Error initializing promo codes: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = init_promo_codes()
    if success:
        print("\n🎉 Promo code initialization completed successfully!")
    else:
        print("\n❌ Promo code initialization failed!")
        sys.exit(1) 
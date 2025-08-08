"""
Promo Code Management System
"""

import logging
from datetime import datetime, timedelta
from flask import current_app

logger = logging.getLogger(__name__)

# Define promo codes
PROMO_CODES = {
    'EARLYZYPPTS': {
        'description': 'Early Access - Pro features with free credits',
        'early_access': True,
        'bonus_credits': 0,  # No bonus credits, just early access
        'plan_upgrade': None,  # Stays on free plan
        'max_uses': None,  # Unlimited uses
        'expires_at': None,  # No expiration
        'discount_percentage': 0,
        'discount_amount': 0.0
    }
}

def validate_promo_code(code):
    """
    Validate a promo code and return its details if valid
    """
    if not code:
        return None
    
    code = code.upper().strip()
    
    # Check if code exists in our predefined codes
    if code in PROMO_CODES:
        promo_details = PROMO_CODES[code]
        
        # Check if code is active (no expiration or not expired)
        if promo_details.get('expires_at'):
            if datetime.utcnow() > promo_details['expires_at']:
                return None
        
        return {
            'code': code,
            'valid': True,
            **promo_details
        }
    
    # Check database for custom promo codes
    try:
        # Try relative import first
        try:
            from ..models import PromoCode
            from ..app_config import db
        except ImportError:
            # Try absolute import
            try:
                from models import PromoCode
                from app_config import db
            except ImportError:
                logger.error("Could not import models for promo code validation")
                return None
        
        promo_code = PromoCode.query.filter_by(code=code).first()
        if promo_code and promo_code.is_valid():
            return {
                'code': promo_code.code,
                'valid': True,
                'description': promo_code.description,
                'early_access': promo_code.early_access,
                'bonus_credits': promo_code.bonus_credits,
                'plan_upgrade': promo_code.plan_upgrade,
                'discount_percentage': promo_code.discount_percentage,
                'discount_amount': promo_code.discount_amount
            }
    except Exception as e:
        logger.error(f"Error checking database promo codes: {e}")
    
    return None

def apply_promo_code(user, promo_code):
    """
    Apply a promo code to a user
    """
    if not user or not promo_code:
        return False, "Invalid user or promo code"
    
    try:
        # Validate the promo code
        promo_details = validate_promo_code(promo_code)
        if not promo_details:
            return False, "Invalid or expired promo code"
        
        # Check if user already used a promo code
        if user.promo_code_applied:
            return False, "User has already used a promo code"
        
        # Apply the promo code
        user.promo_code_used = promo_details['code']
        user.promo_code_applied = True
        
        # Handle early access
        if promo_details.get('early_access'):
            logger.info(f"Applied early access promo code {promo_details['code']} to user {user.id}")
        
        # Handle bonus credits
        if promo_details.get('bonus_credits', 0) > 0:
            if user.subscription:
                user.subscription.monthly_credits += promo_details['bonus_credits']
                logger.info(f"Added {promo_details['bonus_credits']} bonus credits to user {user.id}")
        
        # Handle plan upgrade
        if promo_details.get('plan_upgrade'):
            if user.subscription:
                user.subscription.plan = promo_details['plan_upgrade']
                logger.info(f"Upgraded user {user.id} to {promo_details['plan_upgrade']} plan")
        
        # Save changes
        try:
            from ..app_config import db
        except ImportError:
            try:
                from app_config import db
            except ImportError:
                logger.error("Could not import db for saving promo code")
                return False, "Database error"
        
        db.session.commit()
        
        return True, f"Promo code {promo_details['code']} applied successfully"
        
    except Exception as e:
        logger.error(f"Error applying promo code: {e}")
        return False, f"Error applying promo code: {str(e)}"

def create_early_zyppts_promo_code():
    """
    Create the EARLYZYPPTS promo code in the database if it doesn't exist
    """
    try:
        # Try relative import first
        try:
            from ..models import PromoCode
            from ..app_config import db
        except ImportError:
            # Try absolute import
            try:
                from models import PromoCode
                from app_config import db
            except ImportError:
                logger.error("Could not import models for creating promo code")
                return None
        
        # Check if promo code already exists
        existing_code = PromoCode.query.filter_by(code='EARLYZYPPTS').first()
        if existing_code:
            logger.info("EARLYZYPPTS promo code already exists")
            return existing_code
        
        # Create the promo code
        promo_code = PromoCode(
            code='EARLYZYPPTS',
            description='Early Access - Pro features with free credits',
            early_access=True,
            bonus_credits=0,
            plan_upgrade=None,
            max_uses=None,
            is_active=True,
            expires_at=None
        )
        
        db.session.add(promo_code)
        db.session.commit()
        
        logger.info("EARLYZYPPTS promo code created successfully")
        return promo_code
        
    except Exception as e:
        logger.error(f"Error creating EARLYZYPPTS promo code: {e}")
        return None

def get_user_promo_code_status(user):
    """
    Get the promo code status for a user
    """
    if not user:
        return None
    
    if not user.promo_code_applied:
        return None
    
    return {
        'code': user.promo_code_used,
        'applied': user.promo_code_applied,
        'early_access': user.has_early_access(),
        'features': user.get_available_features()
    } 
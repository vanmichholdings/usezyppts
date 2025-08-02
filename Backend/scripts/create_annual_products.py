#!/usr/bin/env python3
"""
Script to create annual subscription products in Stripe
This script helps set up the annual pricing products that are needed for the subscription system.
"""

import stripe
import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def create_annual_products():
    """
    Create annual subscription products in Stripe
    """
    # Initialize Stripe with the secret key
    stripe.api_key = Config.STRIPE_SECRET_KEY
    
    if not stripe.api_key:
        print("❌ Error: STRIPE_SECRET_KEY not found in environment variables")
        print("Please set your Stripe secret key before running this script")
        return False
    
    print("🔧 Creating annual subscription products in Stripe...")
    print(f"Using Stripe account: {stripe.api_key[:12]}...")
    print()
    
    # Annual pricing configuration
    annual_products = {
        'pro': {
            'name': 'Pro Plan - Annual',
            'description': 'Pro plan billed annually at $7.99/month ($95.90/year)',
            'price': 799,  # $7.99 in cents
            'currency': 'usd',
            'interval': 'month',
            'interval_count': 1,
            'billing_cycle': 'annual'
        },
        'studio': {
            'name': 'Studio Plan - Annual',
            'description': 'Studio plan billed annually at $23/month ($276/year)',
            'price': 2300,  # $23.00 in cents
            'currency': 'usd',
            'interval': 'month',
            'interval_count': 1,
            'billing_cycle': 'annual'
        }
    }
    
    created_products = {}
    
    for plan_key, product_config in annual_products.items():
        try:
            print(f"📦 Creating {product_config['name']}...")
            
            # Create the product
            product = stripe.Product.create(
                name=product_config['name'],
                description=product_config['description'],
                metadata={
                    'plan_type': plan_key,
                    'billing_cycle': 'annual',
                    'created_by': 'annual_product_script',
                    'created_at': datetime.utcnow().isoformat()
                }
            )
            
            # Create the price for the product
            price = stripe.Price.create(
                product=product.id,
                unit_amount=product_config['price'],
                currency=product_config['currency'],
                recurring={
                    'interval': product_config['interval'],
                    'interval_count': product_config['interval_count']
                },
                metadata={
                    'plan_type': plan_key,
                    'billing_cycle': 'annual',
                    'created_by': 'annual_product_script'
                }
            )
            
            created_products[plan_key] = {
                'product_id': product.id,
                'price_id': price.id,
                'name': product_config['name']
            }
            
            print(f"✅ Created {product_config['name']}")
            print(f"   Product ID: {product.id}")
            print(f"   Price ID: {price.id}")
            print()
            
        except stripe.error.StripeError as e:
            print(f"❌ Error creating {product_config['name']}: {str(e)}")
            return False
    
    # Generate the config update
    print("🔧 Generated configuration update:")
    print()
    print("Update your Backend/config.py file with these annual price IDs:")
    print()
    
    for plan_key, product_info in created_products.items():
        print(f"# {product_info['name']}")
        print(f"'stripe_annual_price_id': '{product_info['price_id']}',")
        print()
    
    # Create a backup of the current config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.py')
    backup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config_backup_before_annual_update.py')
    
    try:
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(config_content)
        
        print(f"💾 Backup created: {backup_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create backup: {str(e)}")
    
    print("🎉 Annual products created successfully!")
    print("📝 Next steps:")
    print("1. Update the stripe_annual_price_id values in Backend/config.py")
    print("2. Test the annual subscription flow")
    print("3. Verify the pricing is correct in your Stripe dashboard")
    
    return True

def list_existing_products():
    """
    List existing products in Stripe to help identify what's already there
    """
    stripe.api_key = Config.STRIPE_SECRET_KEY
    
    if not stripe.api_key:
        print("❌ Error: STRIPE_SECRET_KEY not found in environment variables")
        return False
    
    print("📋 Listing existing Stripe products...")
    print()
    
    try:
        products = stripe.Product.list(limit=50)
        
        if not products.data:
            print("No products found in Stripe account")
            return True
        
        for product in products.data:
            print(f"Product: {product.name}")
            print(f"  ID: {product.id}")
            print(f"  Description: {product.description}")
            print(f"  Metadata: {product.metadata}")
            print()
            
            # Get prices for this product
            prices = stripe.Price.list(product=product.id)
            for price in prices.data:
                print(f"  Price: {price.id}")
                print(f"    Amount: ${price.unit_amount/100:.2f} {price.currency.upper()}")
                if price.recurring:
                    print(f"    Billing: {price.recurring.interval}ly")
                print(f"    Metadata: {price.metadata}")
                print()
        
    except stripe.error.StripeError as e:
        print(f"❌ Error listing products: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Stripe annual subscription products")
    parser.add_argument("--list", action="store_true", help="List existing products")
    parser.add_argument("--create", action="store_true", help="Create annual products")
    
    args = parser.parse_args()
    
    if args.list:
        list_existing_products()
    elif args.create:
        create_annual_products()
    else:
        print("Please specify an action:")
        print("  --list   : List existing Stripe products")
        print("  --create : Create annual subscription products")
        print()
        print("Example: python create_annual_products.py --create") 
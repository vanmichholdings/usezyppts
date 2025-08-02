#!/usr/bin/env python3
"""
Script to update Stripe API keys and test configuration
This script helps fix expired Stripe API keys and verify the configuration.
"""

import os
import sys
import stripe
from datetime import datetime

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_stripe_connection(api_key):
    """
    Test if a Stripe API key is valid
    """
    try:
        stripe.api_key = api_key
        # Try to list products to test the connection
        products = stripe.Product.list(limit=1)
        return True, "API key is valid"
    except stripe.error.AuthenticationError:
        return False, "Invalid API key"
    except stripe.error.APIError as e:
        return False, f"API Error: {str(e)}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def update_stripe_keys():
    """
    Interactive script to update Stripe API keys
    """
    print("🔧 Stripe API Key Update Tool")
    print("=" * 50)
    print()
    
    # Check current environment
    current_secret = os.environ.get('STRIPE_SECRET_KEY')
    current_publishable = os.environ.get('STRIPE_PUBLISHABLE_KEY')
    current_webhook = os.environ.get('STRIPE_WEBHOOK_SECRET')
    
    print("📋 Current Configuration:")
    if current_secret:
        print(f"  Secret Key: {current_secret[:12]}...{current_secret[-4:]}")
    else:
        print("  Secret Key: Not set")
    
    if current_publishable:
        print(f"  Publishable Key: {current_publishable[:12]}...{current_publishable[-4:]}")
    else:
        print("  Publishable Key: Not set")
    
    if current_webhook:
        print(f"  Webhook Secret: {current_webhook[:12]}...{current_webhook[-4:]}")
    else:
        print("  Webhook Secret: Not set")
    
    print()
    
    # Test current keys if they exist
    if current_secret:
        print("🧪 Testing current secret key...")
        is_valid, message = test_stripe_connection(current_secret)
        if is_valid:
            print(f"  ✅ {message}")
        else:
            print(f"  ❌ {message}")
            print("  🔄 Key appears to be expired or invalid")
        print()
    
    # Get new keys
    print("🔄 Enter new Stripe API keys:")
    print()
    
    new_secret = input("Enter new STRIPE_SECRET_KEY (or press Enter to skip): ").strip()
    new_publishable = input("Enter new STRIPE_PUBLISHABLE_KEY (or press Enter to skip): ").strip()
    new_webhook = input("Enter new STRIPE_WEBHOOK_SECRET (or press Enter to skip): ").strip()
    
    print()
    
    # Test new secret key if provided
    if new_secret:
        print("🧪 Testing new secret key...")
        is_valid, message = test_stripe_connection(new_secret)
        if is_valid:
            print(f"  ✅ {message}")
            
            # Show account information
            try:
                account = stripe.Account.retrieve()
                print(f"  📊 Account: {account.business_profile.name or 'N/A'}")
                print(f"  🌍 Country: {account.country}")
                print(f"  💰 Currency: {account.default_currency}")
            except Exception as e:
                print(f"  ⚠️  Could not retrieve account info: {str(e)}")
        else:
            print(f"  ❌ {message}")
            print("  ⚠️  Please check your API key and try again")
            return False
        print()
    
    # Update environment variables
    updated = False
    
    if new_secret:
        os.environ['STRIPE_SECRET_KEY'] = new_secret
        updated = True
        print("✅ Updated STRIPE_SECRET_KEY")
    
    if new_publishable:
        os.environ['STRIPE_PUBLISHABLE_KEY'] = new_publishable
        updated = True
        print("✅ Updated STRIPE_PUBLISHABLE_KEY")
    
    if new_webhook:
        os.environ['STRIPE_WEBHOOK_SECRET'] = new_webhook
        updated = True
        print("✅ Updated STRIPE_WEBHOOK_SECRET")
    
    if not updated:
        print("ℹ️  No keys were updated")
        return True
    
    print()
    
    # Test the updated configuration
    print("🧪 Testing updated configuration...")
    try:
        from config import Config
        
        if Config.STRIPE_SECRET_KEY:
            is_valid, message = test_stripe_connection(Config.STRIPE_SECRET_KEY)
            if is_valid:
                print(f"  ✅ Configuration test passed: {message}")
            else:
                print(f"  ❌ Configuration test failed: {message}")
                return False
        else:
            print("  ⚠️  No secret key found in configuration")
        
        # Test listing products
        products = stripe.Product.list(limit=5)
        print(f"  📦 Found {len(products.data)} products in your Stripe account")
        
        # Test listing prices
        prices = stripe.Price.list(limit=5)
        print(f"  💰 Found {len(prices.data)} prices in your Stripe account")
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {str(e)}")
        return False
    
    print()
    print("🎉 Stripe configuration updated successfully!")
    print()
    print("📝 Next steps:")
    print("1. Update your environment variables in production")
    print("2. Restart your application")
    print("3. Test the subscription flow")
    print("4. Verify webhook endpoints are working")
    
    return True

def list_stripe_products():
    """
    List all products and prices in Stripe
    """
    try:
        from config import Config
        
        if not Config.STRIPE_SECRET_KEY:
            print("❌ No Stripe secret key configured")
            return False
        
        stripe.api_key = Config.STRIPE_SECRET_KEY
        
        print("📋 Stripe Products and Prices")
        print("=" * 50)
        print()
        
        # List products
        products = stripe.Product.list(limit=50)
        
        if not products.data:
            print("No products found in Stripe account")
            return True
        
        for product in products.data:
            print(f"📦 Product: {product.name}")
            print(f"  ID: {product.id}")
            print(f"  Description: {product.description}")
            print(f"  Active: {product.active}")
            print(f"  Metadata: {product.metadata}")
            print()
            
            # Get prices for this product
            prices = stripe.Price.list(product=product.id, active=True)
            for price in prices.data:
                print(f"  💰 Price: {price.id}")
                print(f"    Amount: ${price.unit_amount/100:.2f} {price.currency.upper()}")
                if price.recurring:
                    print(f"    Billing: {price.recurring.interval}ly")
                    if price.recurring.interval_count > 1:
                        print(f"    Interval Count: {price.recurring.interval_count}")
                print(f"    Active: {price.active}")
                print(f"    Metadata: {price.metadata}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing products: {str(e)}")
        return False

def create_env_file():
    """
    Create a .env file template with Stripe configuration
    """
    print("📝 Creating .env file template...")
    print()
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Check if .env file already exists
    if os.path.exists(env_path):
        print(f"⚠️  .env file already exists at: {env_path}")
        print("📋 Current content:")
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"❌ Could not read existing .env file: {str(e)}")
        
        print()
        response = input("Do you want to overwrite the existing .env file? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Operation cancelled. Existing .env file preserved.")
            return False
        
        # Create backup
        backup_path = env_path + '.backup'
        try:
            with open(env_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            print(f"💾 Backup created at: {backup_path}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {str(e)}")
    
    env_content = """# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_your_new_secret_key_here
STRIPE_PUBLISHABLE_KEY=pk_live_your_new_publishable_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_new_webhook_secret_here

# Other Configuration
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Created .env file at: {env_path}")
        print("📝 Please update the values with your actual API keys")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update and test Stripe API keys")
    parser.add_argument("--update", action="store_true", help="Update Stripe API keys")
    parser.add_argument("--list", action="store_true", help="List Stripe products")
    parser.add_argument("--env", action="store_true", help="Create .env file template")
    
    args = parser.parse_args()
    
    if args.update:
        update_stripe_keys()
    elif args.list:
        list_stripe_products()
    elif args.env:
        create_env_file()
    else:
        print("Please specify an action:")
        print("  --update : Update Stripe API keys")
        print("  --list   : List Stripe products")
        print("  --env    : Create .env file template")
        print()
        print("Example: python update_stripe_keys.py --update") 
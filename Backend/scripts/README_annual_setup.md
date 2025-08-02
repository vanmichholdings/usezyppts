# Annual Subscription Setup Guide

This guide explains how to set up annual subscription products in Stripe and configure them in the application.

## Overview

The application currently has monthly subscription products configured, but users selecting annual billing are being charged the monthly rate. This setup will create proper annual subscription products with the correct pricing:

- **Pro Plan Annual**: $7.99/month billed annually ($95.90/year)
- **Studio Plan Annual**: $23/month billed annually ($276/year)

## Prerequisites

1. **Stripe Account**: You need access to your Stripe dashboard
2. **Stripe API Keys**: Ensure your `STRIPE_SECRET_KEY` is set in your environment
3. **Python Environment**: Make sure you have the required dependencies installed

## Setup Steps

### Step 1: List Existing Products (Optional)

First, you can check what products already exist in your Stripe account:

```bash
cd Backend/scripts
python create_annual_products.py --list
```

This will show you all existing products and their pricing configurations.

### Step 2: Create Annual Products

Run the script to create the annual subscription products:

```bash
cd Backend/scripts
python create_annual_products.py --create
```

This script will:
- Create new products in Stripe for annual billing
- Set up the correct pricing ($7.99/month for Pro, $23/month for Studio)
- Generate the new product IDs
- Create a backup of your current configuration

### Step 3: Update Configuration

After running the script, you'll see output like this:

```
🔧 Generated configuration update:

Update your Backend/config.py file with these annual price IDs:

# Pro Plan - Annual
'stripe_annual_price_id': 'price_1ABC123DEF456...',

# Studio Plan - Annual
'stripe_annual_price_id': 'price_1XYZ789GHI012...',
```

Copy these new price IDs and update your `Backend/config.py` file:

```python
'pro': {
    'price': 9.99,
    'stripe_price_id': 'price_1RnCQDI1902kkwjouP5vvijE',
    'stripe_annual_price_id': 'price_1ABC123DEF456...',  # Replace with actual ID
    # ... rest of config
},
'studio': {
    'price': 29.99,
    'stripe_price_id': 'price_1RnCRWI1902kkwjoq18LY3eB',
    'stripe_annual_price_id': 'price_1XYZ789GHI012...',  # Replace with actual ID
    # ... rest of config
},
```

### Step 4: Test the Implementation

1. **Deploy the changes** to your production environment
2. **Test the subscription flow**:
   - Go to the subscription plans page
   - Toggle to annual billing
   - Select a plan and proceed to checkout
   - Verify the correct annual pricing is displayed
3. **Check Stripe Dashboard**:
   - Verify the new products are created
   - Confirm the pricing is correct
   - Test a subscription creation

## Configuration Details

### Current Monthly Pricing
- **Pro**: $9.99/month
- **Studio**: $29.99/month

### New Annual Pricing
- **Pro**: $7.99/month billed annually ($95.90/year) - 20% savings
- **Studio**: $23/month billed annually ($276/year) - 23% savings

### Frontend Integration

The frontend already supports annual billing with the correct pricing display:
- Pro: $7.99/month billed annually
- Studio: $23/month billed annually
- Enterprise: $79.20/month billed annually

## Troubleshooting

### Common Issues

1. **"STRIPE_SECRET_KEY not found"**
   - Ensure your Stripe secret key is set in your environment variables
   - Check that the key is valid and has the correct permissions

2. **"Product already exists"**
   - The script will handle this gracefully
   - You can use the `--list` option to see existing products

3. **Incorrect pricing in checkout**
   - Verify the price IDs are correctly updated in the config
   - Check that the Stripe products have the correct pricing set

### Verification Steps

1. **Check Stripe Dashboard**:
   - Products → Verify annual products exist
   - Pricing → Confirm amounts are correct
   - Metadata → Check that billing_cycle is set to 'annual'

2. **Test Subscription Flow**:
   - Annual toggle should show correct pricing
   - Checkout should use annual product IDs
   - Billing should reflect annual amounts

## Backup and Rollback

The script automatically creates a backup of your configuration before making changes:
- Backup location: `Backend/config_backup_before_annual_update.py`
- To rollback: Restore the backup file and restart the application

## Support

If you encounter any issues:
1. Check the Stripe dashboard for product creation status
2. Verify environment variables are correctly set
3. Review the application logs for any error messages
4. Test with a small amount first to verify the setup

## Security Notes

- Never commit Stripe secret keys to version control
- Use environment variables for all sensitive configuration
- Regularly rotate your Stripe API keys
- Monitor your Stripe dashboard for any unusual activity 
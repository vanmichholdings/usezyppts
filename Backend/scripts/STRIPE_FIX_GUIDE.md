# 🔧 Stripe API Key Fix Guide

## 🚨 Issue: Expired Stripe API Key

Your Stripe API key has expired, causing subscription payments to fail.

## ⚠️ Important Note

**The .env file was overwritten** with placeholder values. You need to restore your actual API keys.

## ✅ Quick Fix Steps

### Step 1: Get Your Actual API Keys

**Option A: From Render Dashboard (Recommended)**
1. **Go to your Render Dashboard**: https://dashboard.render.com/
2. **Select your web service**
3. **Go to Environment → Environment Variables**
4. **Copy the actual values** for:
   - `STRIPE_SECRET_KEY`
   - `STRIPE_PUBLISHABLE_KEY`
   - `STRIPE_WEBHOOK_SECRET`

**Option B: From Stripe Dashboard**
1. **Log into your Stripe Dashboard**: https://dashboard.stripe.com/
2. **Go to Developers → API Keys**
3. **Copy your current API keys**:
   - **Secret Key**: `sk_live_...` (for server-side)
   - **Publishable Key**: `pk_live_...` (for client-side)
   - **Webhook Secret**: `whsec_...` (for webhook verification)

### Step 2: Update Your .env File

**Replace the placeholder values** in `Backend/.env` with your actual keys:

```env
# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_your_actual_secret_key_here
STRIPE_PUBLISHABLE_KEY=pk_live_your_actual_publishable_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_actual_webhook_secret_here

# Other Configuration
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

### Step 3: Test the Configuration

```bash
cd Backend/scripts
python update_stripe_keys.py --list
```

This will verify your keys work and show your products.

### Step 4: Restart Your Application

**Local Development:**
```bash
# Stop your current Flask app (Ctrl+C)
# Then restart it
cd Backend
python run.py
```

**Production (Render):**
- Render will automatically restart when you update environment variables

## 🔍 Verification

After updating, test that subscriptions work:

1. **Go to your subscription page**
2. **Try to create a test subscription**
3. **Check the logs for any errors**
4. **Verify in Stripe Dashboard** that subscriptions are created

## 📋 Required API Keys

Make sure you have all three keys:

1. **STRIPE_SECRET_KEY** - For server-side operations
2. **STRIPE_PUBLISHABLE_KEY** - For client-side checkout
3. **STRIPE_WEBHOOK_SECRET** - For webhook verification

## 🚨 Important Notes

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Test with small amounts** first
- **Monitor your Stripe dashboard** for any issues
- **Always backup existing files** before making changes

## 🆘 If You Need Help

1. **Check Stripe Dashboard** for API key status
2. **Verify webhook endpoints** are still configured
3. **Test with the verification script** provided
4. **Check application logs** for specific error messages

## ✅ Success Indicators

- ✅ No more "Expired API Key" errors
- ✅ Subscription checkout works
- ✅ Webhooks are received
- ✅ Subscriptions appear in Stripe Dashboard

## 🔧 Prevention

The script has been updated to:
- ✅ Check if .env file exists before overwriting
- ✅ Create backups automatically
- ✅ Ask for confirmation before overwriting
- ✅ Show current content before making changes 
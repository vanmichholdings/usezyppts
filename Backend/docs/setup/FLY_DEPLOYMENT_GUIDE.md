# Fly.io Deployment Guide for Zyppts Logo Processor

## 🚀 Quick Deployment

### 1. Install Fly CLI
```bash
# macOS
brew install flyctl

# Or download from https://fly.io/docs/hands-on/install-flyctl/
```

### 2. Login to Fly.io
```bash
flyctl auth login
```

### 3. Run Deployment Script
```bash
# Navigate to Backend directory
cd Backend
./deploy_fly.sh
```

## 🔧 Manual Deployment Steps

### 1. Create App
```bash
flyctl apps create zyppts-logo-processor
```

### 2. Set Secrets
```bash
flyctl secrets set SECRET_KEY="your-secret-key-here"
flyctl secrets set MAIL_USERNAME="zyppts@gmail.com"
flyctl secrets set MAIL_PASSWORD="your-app-password"
flyctl secrets set STRIPE_SECRET_KEY="your-stripe-secret"
flyctl secrets set STRIPE_PUBLISHABLE_KEY="your-stripe-publishable"
flyctl secrets set STRIPE_WEBHOOK_SECRET="your-webhook-secret"
flyctl secrets set ADMIN_ALERT_EMAIL="mike@usezyppts.com,zyppts@gmail.com"
```

### 3. Set up Database
```bash
flyctl postgres create zyppts-db
flyctl postgres attach zyppts-db --app zyppts-logo-processor
```

### 4. Set up Redis
```bash
flyctl redis create zyppts-redis
flyctl redis attach zyppts-redis --app zyppts-logo-processor
```

### 5. Create Volume
```bash
flyctl volumes create zyppts_data --size 3 --region iad
```

### 6. Deploy
```bash
# Deploy from root directory
cd ..
flyctl deploy
```

## 📊 Monitoring

### Check Status
```bash
flyctl status
```

### View Logs
```bash
flyctl logs
```

### Open App
```bash
flyctl open
```

## 💰 Cost Optimization

### Free Tier Limits
- 3 shared-cpu-1x 256mb VMs (free)
- 3GB persistent volume storage (free)
- 160GB outbound data transfer (free)

### Recommended Configuration
```bash
# Scale to 1 VM to stay within free tier
flyctl scale count 1

# Use shared CPU with 512MB RAM
flyctl scale vm shared-cpu-1x --memory 512
```

## 🔧 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   flyctl logs
   # Check for dependency issues
   ```

2. **Memory Issues**
   ```bash
   # Scale up memory
   flyctl scale vm shared-cpu-1x --memory 1024
   ```

3. **Database Connection Issues**
   ```bash
   # Check database status
   flyctl postgres list
   flyctl postgres connect zyppts-db
   ```

4. **Redis Connection Issues**
   ```bash
   # Check Redis status
   flyctl redis list
   flyctl redis connect zyppts-redis
   ```

## 📁 File Structure

```
zyppts_v10/
├── fly.toml              # Fly.io configuration
├── Dockerfile            # Container configuration
├── .dockerignore         # Docker ignore file
├── requirements.txt      # Optimized Python dependencies
├── deploy_fly.sh         # Deployment script
└── Backend/
    ├── gunicorn.conf.py  # Gunicorn configuration
    └── ...               # Application code
```

## 🌐 Environment Variables

The following environment variables are automatically set by Fly.io:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `PORT` - Application port (8080)

## 🔒 Security

- HTTPS is automatically enabled
- Secrets are encrypted and secure
- Non-root user in container
- Health checks enabled

## 📈 Scaling

### Scale Up (Paid)
```bash
flyctl scale count 2
flyctl scale vm shared-cpu-2x --memory 1024
```

### Scale Down (Free)
```bash
flyctl scale count 1
flyctl scale vm shared-cpu-1x --memory 512
```

## 🎯 Success Indicators

✅ App is accessible at `https://zyppts-logo-processor.fly.dev`
✅ Health check passes: `https://zyppts-logo-processor.fly.dev/health`
✅ Database connection working
✅ Redis connection working
✅ File uploads working
✅ Logo processing functional

## 📞 Support

If you encounter issues:
1. Check logs: `flyctl logs`
2. Check status: `flyctl status`
3. Restart app: `flyctl restart`
4. View Fly.io documentation: https://fly.io/docs/

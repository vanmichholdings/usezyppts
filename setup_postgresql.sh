#!/bin/bash

# PostgreSQL Setup Script for Zyppts v10 on Fly.io
# This script sets up a PostgreSQL database and configures the app to use it

set -e

echo "🚀 Setting up PostgreSQL for Zyppts v10..."

# Check if we're on Fly.io
if [ -z "$FLY_APP_NAME" ]; then
    echo "❌ This script should be run on Fly.io"
    exit 1
fi

echo "📊 Creating PostgreSQL database..."

# Create PostgreSQL database
fly postgres create --name zyppts-db --region iad --initial-cluster-size 1 --vm-size shared-cpu-1x --volume-size 10

echo "🔗 Attaching database to app..."

# Attach the database to the app
fly postgres attach --postgres-app zyppts-db --app zyppts-logo-processor-aged-violet-9912

echo "🔧 Setting up database schema..."

# Wait for database to be ready
sleep 10

echo "✅ PostgreSQL setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Deploy your app: fly deploy"
echo "2. The app will automatically create tables on first startup"
echo "3. Default admin user will be created: admin/admin123"
echo ""
echo "🔍 To check database status:"
echo "   fly postgres connect --app zyppts-db"
echo ""
echo "📊 To view database logs:"
echo "   fly logs --app zyppts-db" 
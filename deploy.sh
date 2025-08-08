#!/bin/bash

# Zyppts v10 Deployment Script for Fly.io
# This script sets up PostgreSQL and deploys the application

set -e

echo "🚀 Zyppts v10 Deployment Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if fly CLI is installed
if ! command -v fly &> /dev/null; then
    print_error "Fly CLI is not installed. Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if we're logged in to Fly.io
if ! fly auth whoami &> /dev/null; then
    print_error "Not logged in to Fly.io. Please run: fly auth login"
    exit 1
fi

print_status "Checking current app status..."

# Check if app exists
if fly apps list | grep -q "zyppts-logo-processor-aged-violet-9912"; then
    print_success "App exists on Fly.io"
else
    print_error "App not found. Please create it first or check the app name in fly.toml"
    exit 1
fi

# Check if PostgreSQL database exists
print_status "Checking PostgreSQL database..."

if fly postgres list | grep -q "zyppts-db"; then
    print_success "PostgreSQL database exists"
    DB_EXISTS=true
else
    print_warning "PostgreSQL database not found. Creating new database..."
    DB_EXISTS=false
fi

# Create PostgreSQL database if it doesn't exist
if [ "$DB_EXISTS" = false ]; then
    print_status "Creating PostgreSQL database..."
    
    # Create the database
    fly postgres create --name zyppts-db --region iad --initial-cluster-size 1 --vm-size shared-cpu-1x --volume-size 10
    
    if [ $? -eq 0 ]; then
        print_success "PostgreSQL database created successfully"
    else
        print_error "Failed to create PostgreSQL database"
        exit 1
    fi
    
    # Wait a moment for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 15
fi

# Attach database to app
print_status "Attaching database to app..."
fly postgres attach zyppts-db --app zyppts-logo-processor-aged-violet-9912

if [ $? -eq 0 ]; then
    print_success "Database attached successfully"
else
    print_warning "Database attachment failed or already attached"
fi

# Deploy the application
print_status "Deploying application..."
fly deploy

if [ $? -eq 0 ]; then
    print_success "Application deployed successfully!"
else
    print_error "Deployment failed"
    exit 1
fi

# Wait for deployment to be ready
print_status "Waiting for deployment to be ready..."
sleep 30

# Check app status
print_status "Checking app status..."
fly status

# Show logs
print_status "Recent logs:"
fly logs --app zyppts-logo-processor-aged-violet-9912 --limit 20

print_success "Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Visit your app: https://zyppts-logo-processor-aged-violet-9912.fly.dev"
echo "2. Login with default credentials:"
echo "   - Username: admin"
echo "   - Password: admin123"
echo "   - Or username: mike, password: admin123"
echo ""
echo "🔍 Useful commands:"
echo "   View logs: fly logs --app zyppts-logo-processor-aged-violet-9912"
echo "   Check status: fly status"
echo "   Connect to database: fly postgres connect --app zyppts-db"
echo "   Scale app: fly scale count 1"
echo ""
print_success "🎉 Zyppts v10 is now running on Fly.io with PostgreSQL!" 
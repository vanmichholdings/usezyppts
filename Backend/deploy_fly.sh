#!/bin/bash

# Fly.io Deployment Script for Zyppts v10
# This script deploys your app to Fly.io with optimized settings

set -e  # Exit on any error

echo "🚀 Starting Fly.io deployment for Zyppts v10..."

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

# Check if we're in the right directory
check_directory() {
    if [ ! -f "../Dockerfile" ] || [ ! -f "../fly.toml" ]; then
        print_error "This script must be run from the Backend directory"
        exit 1
    fi
}

# Check if Fly CLI is installed
check_fly_cli() {
    print_status "Checking Fly CLI installation..."
    if ! command -v fly &> /dev/null; then
        print_error "Fly CLI is not installed. Installing now..."
        curl -L https://fly.io/install.sh | sh
        export PATH="$HOME/.fly/bin:$PATH"
        print_success "Fly CLI installed successfully"
    else
        print_success "Fly CLI is already installed"
    fi
}

# Check if user is logged in to Fly.io
check_fly_auth() {
    print_status "Checking Fly.io authentication..."
    if ! fly auth whoami &> /dev/null; then
        print_warning "Not logged in to Fly.io. Please log in..."
        fly auth login
    else
        print_success "Already authenticated with Fly.io"
    fi
}

# Create Fly.io app if it doesn't exist
create_fly_app() {
    print_status "Checking if Fly.io app exists..."
    if ! fly apps list | grep -q "zyppts-logo-processor"; then
        print_status "Creating new Fly.io app..."
        fly apps create zyppts-logo-processor --org personal
        print_success "Fly.io app created successfully"
    else
        print_success "Fly.io app already exists"
    fi
}

# Create volume for persistent storage
create_volume() {
    print_status "Checking if volume exists..."
    if ! fly volumes list | grep -q "zyppts_data"; then
        print_status "Creating persistent volume..."
        fly volumes create zyppts_data --size 10 --region iad
        print_success "Volume created successfully"
    else
        print_success "Volume already exists"
    fi
}

# Set environment variables
set_env_vars() {
    print_status "Setting environment variables..."
    
    # Check if .env file exists
    if [ -f "../.env" ]; then
        print_status "Loading environment variables from .env file..."
        while IFS= read -r line; do
            if [[ $line =~ ^[A-Z_]+=.*$ ]] && [[ ! $line =~ ^# ]]; then
                fly secrets set "$line"
            fi
        done < ../.env
        print_success "Environment variables set from .env file"
    else
        print_warning "No .env file found. Please set environment variables manually:"
        echo "fly secrets set SECRET_KEY=your-secret-key"
        echo "fly secrets set DATABASE_URL=your-database-url"
        echo "fly secrets set REDIS_URL=your-redis-url"
        echo "fly secrets set STRIPE_SECRET_KEY=your-stripe-secret"
        echo "fly secrets set STRIPE_PUBLISHABLE_KEY=your-stripe-publishable"
        echo "fly secrets set MAIL_USERNAME=your-email"
        echo "fly secrets set MAIL_PASSWORD=your-password"
    fi
}

# Deploy the application
deploy_app() {
    print_status "Deploying application to Fly.io..."
    
    # Change to root directory for deployment
    cd ..
    
    # Build and deploy
    fly deploy --remote-only
    
    print_success "Application deployed successfully!"
}

# Check deployment status
check_deployment() {
    print_status "Checking deployment status..."
    
    # Wait a moment for deployment to complete
    sleep 10
    
    # Check app status
    fly status
    
    # Check health endpoint
    print_status "Testing health endpoint..."
    APP_URL=$(fly status --json | jq -r '.Current.IPAddress')
    if curl -f "https://$APP_URL/health" > /dev/null 2>&1; then
        print_success "Health check passed!"
    else
        print_warning "Health check failed. Checking logs..."
        fly logs
    fi
}

# Show deployment information
show_info() {
    print_success "Deployment completed!"
    echo ""
    echo "🌐 Your app is now live at:"
    echo "   https://zyppts-logo-processor.fly.dev"
    echo ""
    echo "📊 Monitor your app:"
    echo "   fly status"
    echo "   fly logs"
    echo "   fly dashboard"
    echo ""
    echo "🔧 Useful commands:"
    echo "   fly scale count 2    # Scale to 2 instances"
    echo "   fly scale memory 2048 # Set memory to 2GB"
    echo "   fly logs -f         # Follow logs in real-time"
    echo "   fly ssh console     # SSH into the app"
    echo ""
    echo "💰 Current pricing:"
    echo "   - 2GB RAM: ~$7.50/month"
    echo "   - 1 CPU: Included"
    echo "   - Storage: 10GB included"
    echo "   - Bandwidth: Included"
}

# Main deployment process
main() {
    echo "🎨 Zyppts v10 - Fly.io Deployment"
    echo "=================================="
    echo ""
    
    # Check directory
    check_directory
    
    # Run deployment steps
    check_fly_cli
    check_fly_auth
    create_fly_app
    create_volume
    set_env_vars
    deploy_app
    check_deployment
    show_info
}

# Run main function
main "$@" 
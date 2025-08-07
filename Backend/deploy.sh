#!/bin/bash

# Zyppts v10 Docker Deployment Script
# This script helps you deploy the optimized Zyppts application

set -e

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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install it and try again."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to build the application
build_app() {
    print_status "Building Zyppts v10 application..."
    docker-compose build --no-cache
    print_success "Application built successfully"
}

# Function to start the application
start_app() {
    print_status "Starting Zyppts v10 application..."
    docker-compose up -d
    print_success "Application started successfully"
}

# Function to start development environment
start_dev() {
    print_status "Starting Zyppts v10 development environment..."
    docker-compose -f docker-compose.dev.yml up -d
    print_success "Development environment started successfully"
}

# Function to stop the application
stop_app() {
    print_status "Stopping Zyppts v10 application..."
    docker-compose down
    print_success "Application stopped successfully"
}

# Function to stop development environment
stop_dev() {
    print_status "Stopping Zyppts v10 development environment..."
    docker-compose -f docker-compose.dev.yml down
    print_success "Development environment stopped successfully"
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f
}

# Function to show development logs
show_dev_logs() {
    print_status "Showing development logs..."
    docker-compose -f docker-compose.dev.yml logs -f
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Function to show status
show_status() {
    print_status "Showing application status..."
    docker-compose ps
}

# Function to restart the application
restart_app() {
    print_status "Restarting Zyppts v10 application..."
    docker-compose restart
    print_success "Application restarted successfully"
}

# Function to show help
show_help() {
    echo "🎨 Zyppts v10 Docker Deployment Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the application"
    echo "  start       - Start the application"
    echo "  stop        - Stop the application"
    echo "  restart     - Restart the application"
    echo "  logs        - Show application logs"
    echo "  status      - Show application status"
    echo "  dev-start   - Start development environment"
    echo "  dev-stop    - Stop development environment"
    echo "  dev-logs    - Show development logs"
    echo "  cleanup     - Clean up Docker resources"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 dev-start"
    echo "  $0 logs"
}

# Main script logic
case "${1:-help}" in
    build)
        check_docker
        check_docker_compose
        build_app
        ;;
    start)
        check_docker
        check_docker_compose
        start_app
        ;;
    stop)
        check_docker
        check_docker_compose
        stop_app
        ;;
    restart)
        check_docker
        check_docker_compose
        restart_app
        ;;
    logs)
        check_docker
        check_docker_compose
        show_logs
        ;;
    status)
        check_docker
        check_docker_compose
        show_status
        ;;
    dev-start)
        check_docker
        check_docker_compose
        start_dev
        ;;
    dev-stop)
        check_docker
        check_docker_compose
        stop_dev
        ;;
    dev-logs)
        check_docker
        check_docker_compose
        show_dev_logs
        ;;
    cleanup)
        check_docker
        check_docker_compose
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 
#!/bin/bash

# Microbiome Analysis App - Podman Compose Runner
# Usage: ./run_podman.sh [start|stop|restart|logs|build|clean]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if podman and podman-compose are installed
check_dependencies() {
    if ! command -v podman &> /dev/null; then
        print_error "Podman is not installed!"
        echo "Install with: brew install podman"
        exit 1
    fi

    # podman compose is built-in to podman 3.0+, no separate install needed

    print_success "Dependencies found"
}

# Check if podman machine is running (macOS/Windows)
check_podman_machine() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! podman machine list | grep -q "Currently running"; then
            print_info "Podman machine is not running. Starting..."
            if ! podman machine list | grep -q "podman-machine-default"; then
                print_info "Creating podman machine..."
                podman machine init --cpus 2 --memory 4096 --disk-size 50
            fi
            podman machine start
            print_success "Podman machine started"
        else
            print_success "Podman machine is running"
        fi
    fi
}

# Start the application
start_app() {
    check_dependencies
    check_podman_machine
    
    print_info "Building and starting the application..."
    podman compose up -d --build
    
    print_success "Application started!"
    print_info "Waiting for application to be healthy (this may take 60 seconds)..."
    
    # Wait for healthcheck
    sleep 10
    for i in {1..12}; do
        if podman inspect mcb-microbiome-app 2>/dev/null | grep -q '"Status": "healthy"'; then
            print_success "Application is healthy and ready!"
            echo ""
            echo "🌐 Access the application at: http://localhost:8080"
            echo ""
            echo "Useful commands:"
            echo "  View logs:    ./run_podman.sh logs"
            echo "  Stop app:     ./run_podman.sh stop"
            echo "  Restart app:  ./run_podman.sh restart"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    
    print_info "Application is starting... Check logs with: ./run_podman.sh logs"
    echo "🌐 Try accessing: http://localhost:8080"
}

# Stop the application
stop_app() {
    print_info "Stopping the application..."
    podman compose down
    print_success "Application stopped"
}

# Restart the application
restart_app() {
    print_info "Restarting the application..."
    stop_app
    start_app
}

# View logs
view_logs() {
    print_info "Showing logs (Ctrl+C to exit)..."
    podman compose logs -f
}

# Build without starting
build_app() {
    check_dependencies
    check_podman_machine
    
    print_info "Building the application..."
    podman compose build --no-cache
    print_success "Build complete"
}

# Clean up everything
clean_app() {
    print_info "Cleaning up containers, images, and volumes..."
    
    # Stop containers
    podman compose down -v 2>/dev/null || true
    
    # Remove images
    podman rmi mcb_website-mcb-app 2>/dev/null || true
    podman rmi localhost/mcb_website-mcb-app 2>/dev/null || true
    
    # Prune system
    print_info "Pruning unused resources..."
    podman system prune -f
    
    print_success "Cleanup complete"
}

# Show status
show_status() {
    print_info "Checking application status..."
    echo ""
    
    if podman ps | grep -q "mcb-microbiome-app"; then
        print_success "Container is running"
        podman ps --filter name=mcb-microbiome-app
        echo ""
        
        # Check health
        health=$(podman inspect mcb-microbiome-app 2>/dev/null | grep -o '"Status": "[^"]*"' | head -1 || echo '"Status": "unknown"')
        echo "Health: $health"
        
        echo ""
        echo "🌐 Application URL: http://localhost:8080"
    else
        print_error "Container is not running"
        echo "Start with: ./run_podman.sh start"
    fi
    
    echo ""
    echo "Podman machine status:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        podman machine list
    else
        echo "N/A (Linux)"
    fi
}

# Main script
case "${1:-}" in
    start)
        start_app
        ;;
    stop)
        stop_app
        ;;
    restart)
        restart_app
        ;;
    logs)
        view_logs
        ;;
    build)
        build_app
        ;;
    clean)
        clean_app
        ;;
    status)
        show_status
        ;;
    *)
        echo "Microbiome Analysis App - Podman Manager"
        echo ""
        echo "Usage: ./run_podman.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start     Build and start the application"
        echo "  stop      Stop the application"
        echo "  restart   Restart the application"
        echo "  logs      View application logs (follow mode)"
        echo "  build     Build the image without starting"
        echo "  clean     Stop and remove all containers, images, and volumes"
        echo "  status    Show current status"
        echo ""
        echo "Examples:"
        echo "  ./run_podman.sh start          # Start the app"
        echo "  ./run_podman.sh logs           # View logs"
        echo "  ./run_podman.sh stop           # Stop the app"
        echo ""
        exit 1
        ;;
esac

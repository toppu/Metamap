#!/bin/bash

# Quick status checker for Podman build

echo "🔍 Checking Podman build status..."
echo ""

# Check if build process is still running
if pgrep -f "podman compose" > /dev/null; then
    echo "✨ Build is IN PROGRESS"
    echo ""
    echo "📊 Last 20 lines of build log:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -20 podman_build.log 2>/dev/null || echo "Log file not found yet..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 Monitor full logs with: tail -f podman_build.log"
else
    echo "✅ Build process COMPLETED (or not started)"
    echo ""
    
    # Check container status
    if podman ps | grep -q "mcb-microbiome-app"; then
        echo "🎉 Container is RUNNING!"
        echo ""
        podman ps --filter name=mcb-microbiome-app
        echo ""
        echo "🌐 Access your app at: http://localhost:8080"
    elif podman ps -a | grep -q "mcb-microbiome-app"; then
        echo "⚠️  Container exists but is NOT running"
        echo ""
        podman ps -a --filter name=mcb-microbiome-app
        echo ""
        echo "📋 Check logs with: podman compose logs"
    else
        echo "❌ Container not found"
        echo ""
        echo "📄 Check build log:"
        tail -30 podman_build.log 2>/dev/null || echo "No log file found"
    fi
fi

echo ""
echo "Available commands:"
echo "  ./status.sh          - Check status again"
echo "  tail -f podman_build.log  - Watch build progress"
echo "  podman compose logs  - View container logs"
echo "  podman compose ps    - Check container status"

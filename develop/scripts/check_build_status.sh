#!/bin/bash
# Quick check of MARS build status

LOG_FILE="/tmp/mars_build.log"

echo "🔍 MARS Build Status Check"
echo "=========================="
echo ""

# Check if log exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No build log found at $LOG_FILE"
    echo ""
    echo "Build may not be running. To start:"
    echo "  cd /home/nalin/Develop/mars/develop"
    echo "  ./scripts/build.sh"
    exit 1
fi

# Get last 20 lines
echo "📋 Last 20 lines of build log:"
echo "----------------------------------------"
tail -n 20 "$LOG_FILE"
echo ""
echo "----------------------------------------"

# Check for errors
ERROR_COUNT=$(grep -i "error\|failed" "$LOG_FILE" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "\n⚠️  Found $ERROR_COUNT potential error(s)"
    echo "Recent errors:"
    grep -i "error\|failed" "$LOG_FILE" | tail -5
fi

# Check for completion
if grep -q "Successfully tagged\|build complete" "$LOG_FILE"; then
    echo -e "\n✅ Build appears to be complete!"
    echo "Next steps:"
    echo "  ./scripts/start.sh    # Start the container"
    echo "  ./scripts/enter.sh    # Enter the container"
fi

# Show current step
echo ""
echo "Current step (last non-empty line):"
tail -n 50 "$LOG_FILE" | grep -v "^$" | tail -1

echo ""
echo "📊 To watch live progress:"
echo "  ./scripts/watch_build.sh"
echo ""
echo "Or manually:"
echo "  tail -f $LOG_FILE"


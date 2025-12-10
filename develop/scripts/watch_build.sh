#!/bin/bash
# Watch MARS container build progress

LOG_FILE="/tmp/mars_build.log"
BUILD_PID_FILE="/tmp/mars_build.pid"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   MARS Container Build Monitor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if build is running
if [ -f "$LOG_FILE" ]; then
    echo -e "${GREEN}✓ Build log found${NC}"
    echo ""
    echo "Press Ctrl+C to exit monitor (build will continue in background)"
    echo ""
    echo -e "${YELLOW}Recent build activity:${NC}"
    echo "----------------------------------------"
    
    # Show last 10 lines and follow
    tail -n 10 "$LOG_FILE"
    echo ""
    echo -e "${YELLOW}Following build progress...${NC}"
    echo "----------------------------------------"
    
    # Follow the log with highlighting
    tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
        # Highlight important messages
        if echo "$line" | grep -q "ERROR\|Failed\|failed"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "Successfully\|DONE\|Complete"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "Step\|Building\|Installing"; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
else
    echo -e "${YELLOW}Build log not found. Checking build status...${NC}"
    echo ""
    
    # Check if docker build is running
    if docker ps -a | grep -q "mars"; then
        echo -e "${GREEN}MARS container found${NC}"
        docker ps -a | grep mars
    else
        echo -e "${RED}No MARS build detected${NC}"
        echo ""
        echo "To start a build, run:"
        echo "  cd /home/nalin/Develop/mars/develop"
        echo "  ./scripts/build.sh"
    fi
fi


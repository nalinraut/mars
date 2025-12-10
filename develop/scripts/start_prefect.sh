#!/bin/bash
# Start Prefect server for local development
# This script sets up logging and runs the Prefect server in the background

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directories
mkdir -p /workspace/data/prefect/storage
mkdir -p /workspace/data/prefect/results
mkdir -p /workspace/logs

# Set Prefect environment variables
export PREFECT_HOME="/workspace/data/prefect"
export PREFECT_API_URL="http://127.0.0.1:4200/api"
export PREFECT_LOGGING_LEVEL="${PREFECT_LOGGING_LEVEL:-INFO}"
export PREFECT_LOGGING_INTERNAL_LEVEL="${PREFECT_LOGGING_INTERNAL_LEVEL:-WARNING}"
export PREFECT_LOCAL_STORAGE_PATH="/workspace/data/prefect/storage"

# Disable telemetry
export PREFECT_SEND_ANONYMOUS_USAGE_STATS="false"

# Check if Prefect server is already running
if pgrep -f "prefect server start" > /dev/null; then
    echo -e "${YELLOW}Prefect server already running${NC}"
    echo "API URL: $PREFECT_API_URL"
    exit 0
fi

echo -e "${GREEN}Starting Prefect server...${NC}"

# Start Prefect server in background
nohup prefect server start \
    --host 127.0.0.1 \
    --port 4200 \
    > /workspace/logs/prefect_server.log 2>&1 &

# Wait for server to start
echo "Waiting for Prefect server to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:4200/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}Prefect server started successfully!${NC}"
        echo "  Dashboard: http://127.0.0.1:4200"
        echo "  API URL: $PREFECT_API_URL"
        echo "  Logs: /workspace/logs/prefect_server.log"
        exit 0
    fi
    sleep 1
done

echo -e "${YELLOW}Warning: Prefect server may not be fully started yet${NC}"
echo "Check logs at: /workspace/logs/prefect_server.log"


#!/bin/bash
# Start MARS container

set -e

cd "$(dirname "$0")/.."

echo "Starting MARS container..."
docker compose up -d mars

echo ""
echo "Waiting for container to be ready..."
sleep 3

# Check if Prefect server should be started
if [ "${START_PREFECT_SERVER:-true}" = "true" ]; then
    echo ""
    echo "Starting Prefect server..."
    docker exec mars-container bash -c "/workspace/scripts/start_prefect.sh" || echo "Note: Prefect server start script not found in container"
fi

echo ""
echo "Container started!"
echo ""
echo "Next steps:"
echo "  ./scripts/enter.sh    # Enter the container"
echo "  ./scripts/logs.sh     # View container logs"
if [ "${START_PREFECT_SERVER:-true}" = "true" ]; then
    echo "  Prefect UI: http://127.0.0.1:4200"
fi
echo ""
echo "Container info:"
docker ps --filter name=mars-container


#!/bin/bash
# Enter MARS container

set -e

cd "$(dirname "$0")/.."

CONTAINER_NAME="mars-container"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' is not running."
    echo ""
    echo "Start it first with: ./scripts/start.sh"
    exit 1
fi

echo "Entering MARS container..."
echo ""
docker exec -it "$CONTAINER_NAME" /bin/bash


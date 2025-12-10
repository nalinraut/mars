#!/bin/bash
# View MARS container logs

set -e

cd "$(dirname "$0")/.."

CONTAINER_NAME="mars-container"

echo "MARS container logs:"
echo ""
docker logs -f "$CONTAINER_NAME"


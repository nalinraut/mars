#!/bin/bash
# Stop MARS container

set -e

cd "$(dirname "$0")/.."

echo "Stopping MARS container..."
docker compose stop mars

echo ""
echo "Container stopped!"


#!/bin/bash
# Rebuild MARS container from scratch

set -e

cd "$(dirname "$0")/.."

echo "Rebuilding MARS container..."
echo ""
echo "This will:"
echo "  1. Stop the container"
echo "  2. Remove the container"
echo "  3. Remove the image"
echo "  4. Rebuild from scratch"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Stop and remove container
echo "Stopping container..."
docker compose stop mars 2>/dev/null || true
docker compose rm -f mars 2>/dev/null || true

# Remove image
echo "Removing image..."
docker rmi mars-mars 2>/dev/null || true

# Rebuild
echo "Rebuilding..."
docker compose build --no-cache mars

echo ""
echo "Rebuild complete!"
echo ""
echo "Start with: ./scripts/start.sh"


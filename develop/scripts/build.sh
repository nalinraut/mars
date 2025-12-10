#!/bin/bash
# Build MARS container

set -e

cd "$(dirname "$0")/.."

echo "🔨 Building MARS container..."
docker compose build mars

echo ""
echo "Build complete!"
echo ""
echo "Next steps:"
echo "  ./scripts/start.sh    # Start the container"
echo "  ./scripts/enter.sh    # Enter the container"


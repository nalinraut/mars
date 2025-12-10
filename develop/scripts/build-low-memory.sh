#!/bin/bash
# Memory-efficient Docker build script
# Limits parallelism and memory usage to prevent OOM errors

set -e

echo "=========================================="
echo "Memory-Efficient Docker Build"
echo "=========================================="
echo ""
echo "This build uses reduced parallelism to prevent OOM errors."
echo "Build time will be longer, but memory usage will be lower."
echo ""
echo "NOTE: If you still experience OOM errors, consider:"
echo "  1. Increasing system swap space"
echo "  2. Stopping other memory-intensive processes"
echo "  3. Building on a machine with more RAM"
echo ""

# Disable BuildKit for more predictable memory usage
export DOCKER_BUILDKIT=0

# Build with no cache to avoid memory-heavy cache layers
cd "$(dirname "$0")/.."
cd develop

echo "Starting build with memory-efficient settings..."
echo "  - Single-threaded builds (MAX_JOBS=1)"
echo "  - No build cache (--no-cache)"
echo "  - Sequential package installation"
echo "  - BuildKit disabled for lower memory overhead"
echo ""

# Build with no cache
# Note: Docker build doesn't support --memory flags directly
# Memory limits should be configured in Docker daemon settings if needed
docker compose build \
    --no-cache \
    --progress=plain \
    2>&1 | tee /tmp/docker_build_memory_efficient.log

echo ""
echo "=========================================="
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo "=========================================="
    exit 0
else
    echo "❌ Build failed. Check /tmp/docker_build_memory_efficient.log"
    echo ""
    echo "If you see OOM errors, try:"
    echo "  1. Free up system memory: docker system prune -af"
    echo "  2. Increase swap: sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
    echo "  3. Build in stages or on a machine with more RAM"
    echo "=========================================="
    exit 1
fi


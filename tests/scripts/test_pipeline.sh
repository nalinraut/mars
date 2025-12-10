#!/bin/bash
# Quick test script to run MARS pipeline on a test image
# Usage: ./test_pipeline.sh [image_path] [--run-until STAGE] [--output-dir DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values
DEFAULT_IMAGE="$PROJECT_ROOT/tests/data/test_image_1.jpg"
RUN_UNTIL="composition"
OUTPUT_DIR="/workspace/visualizations/test_run"

# Parse arguments
TEST_IMAGE="${1:-$DEFAULT_IMAGE}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-until)
            RUN_UNTIL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [image_path] [--run-until STAGE] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

echo "🧪 Testing MARS Pipeline"
echo "========================"

# Check if container is running
if ! docker ps | grep -q mars-container; then
    echo "Starting MARS container..."
    cd "$PROJECT_ROOT/develop"
    ./scripts/start.sh
    sleep 5
fi

# Validate image path
if [ ! -f "$TEST_IMAGE" ]; then
    echo "❌ Error: Test image not found: $TEST_IMAGE"
    echo ""
    echo "Available test images:"
    find "$PROJECT_ROOT/tests" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | head -10
    exit 1
fi

echo ""
echo "Test image: $TEST_IMAGE"
echo "Run until: $RUN_UNTIL"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Convert host path to container path
# If image is in tests/, map it to /workspace/tests/
CONTAINER_IMAGE_PATH="$TEST_IMAGE"
if [[ "$TEST_IMAGE" == "$PROJECT_ROOT/tests/"* ]]; then
    # Remove project root and add /workspace prefix
    CONTAINER_IMAGE_PATH="/workspace${TEST_IMAGE#$PROJECT_ROOT}"
elif [[ "$TEST_IMAGE" == "$PROJECT_ROOT/"* ]]; then
    # For other project files, map to /workspace
    CONTAINER_IMAGE_PATH="/workspace${TEST_IMAGE#$PROJECT_ROOT}"
else
    # Absolute path - try to use as-is or copy to container
    echo "⚠️  Warning: Using absolute path. Make sure it's accessible in container."
    CONTAINER_IMAGE_PATH="$TEST_IMAGE"
fi

# Run pipeline inside container using the mars module
docker exec mars-container bash -c "
    cd /workspace && \
    export PYTHONPATH=/workspace/src:\$PYTHONPATH && \
    python -c \"
import sys
sys.path.insert(0, '/workspace/src')
from mars import run
print('=' * 60)
print('MARS Pipeline Test')
print('=' * 60)
print('Input:', '$CONTAINER_IMAGE_PATH')
print('Output:', '$OUTPUT_DIR')
print('Run until:', '$RUN_UNTIL')
print('=' * 60)
result = run(
    image_path='$CONTAINER_IMAGE_PATH',
    output_dir='$OUTPUT_DIR',
    run_until='$RUN_UNTIL'
)
print('')
print('=' * 60)
print('Pipeline Result:', result.get('status', 'unknown'))
print('Stages completed:', ', '.join(result.get('stages_completed', [])))
print('=' * 60)
if 'detection' in result:
    det = result['detection']
    print('Detections:', det.get('count', 0), 'objects')
if 'segmentation' in result:
    seg = result['segmentation']
    print('Masks generated:', seg.get('mask_count', 0))
if 'reconstruction' in result:
    rec = result['reconstruction']
    print('Objects reconstructed:', len(rec.get('objects', [])))
if 'composition' in result:
    comp = result['composition']
    print('Scene ID:', comp.get('scene_id', 'N/A'))
    print('Scene file:', comp.get('scene_path', 'N/A'))
if 'image_id' in result:
    print('Image ID:', result['image_id'])
    print('Results saved to:', '$OUTPUT_DIR')
\"
"

echo ""
echo "✅ Pipeline test complete!"
echo ""
echo "Results location:"
if [[ "$OUTPUT_DIR" == "/workspace/visualizations"* ]]; then
    # Map container path to host path
    HOST_OUTPUT_DIR="$PROJECT_ROOT/visualizations${OUTPUT_DIR#/workspace/visualizations}"
    echo "  Host: $HOST_OUTPUT_DIR"
else
    echo "  Container: $OUTPUT_DIR"
    echo "  (Check inside container or adjust --output-dir to use /workspace/visualizations/)"
fi


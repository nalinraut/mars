#!/bin/bash
# MARS Checkpoint Initialization Script
# Downloads model checkpoints to host-mounted directory if not present
#
# Usage: ./init_checkpoints.sh [--force]
#   --force: Re-download even if checkpoints exist

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
FORCE_DOWNLOAD="${1:-}"

echo "=============================================="
echo "MARS Checkpoint Initialization"
echo "=============================================="
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Ensure HF token is available
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Warning: No Hugging Face token found."
    echo "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated model access."
    echo ""
fi

# Create checkpoint directories
mkdir -p "$CHECKPOINT_DIR/sam3d"
mkdir -p "$CHECKPOINT_DIR/moge"

# =============================================================================
# SAM 3D Objects Checkpoints (~2GB)
# =============================================================================
SAM3D_MARKER="$CHECKPOINT_DIR/sam3d/checkpoints/pipeline.yaml"

if [ -f "$SAM3D_MARKER" ] && [ "$FORCE_DOWNLOAD" != "--force" ]; then
    echo "✓ SAM 3D Objects checkpoints already present"
else
    echo "Downloading SAM 3D Objects checkpoints from Hugging Face..."
    echo "This may take several minutes (~2GB download)"
    echo ""
    
    # Download to temp directory first
    TMP_DIR="$CHECKPOINT_DIR/sam3d-download"
    rm -rf "$TMP_DIR"
    
    huggingface-cli download \
        --repo-type model \
        --local-dir "$TMP_DIR" \
        --max-workers 4 \
        facebook/sam-3d-objects
    
    # Move checkpoints to correct location
    if [ -d "$TMP_DIR/checkpoints" ]; then
        rm -rf "$CHECKPOINT_DIR/sam3d/checkpoints"
        mv "$TMP_DIR/checkpoints" "$CHECKPOINT_DIR/sam3d/"
        echo "✓ SAM 3D Objects checkpoints downloaded"
    else
        echo "✗ Error: SAM 3D Objects checkpoints not found in download"
        exit 1
    fi
    
    # Cleanup
    rm -rf "$TMP_DIR"
fi

# =============================================================================
# MoGe Checkpoints (~1.5GB)
# =============================================================================
# Check for either model.pt or model.safetensors (model format may vary)
MOGE_MARKER_PT="$CHECKPOINT_DIR/moge/model.pt"
MOGE_MARKER_SAFE="$CHECKPOINT_DIR/moge/model.safetensors"

if ([ -f "$MOGE_MARKER_PT" ] || [ -f "$MOGE_MARKER_SAFE" ]) && [ "$FORCE_DOWNLOAD" != "--force" ]; then
    echo "✓ MoGe checkpoints already present"
else
    echo "Downloading MoGe checkpoints from Hugging Face..."
    echo "This may take several minutes (~1.5GB download)"
    echo ""
    
    # Download MoGe model (ViT-Large variant)
    huggingface-cli download \
        --repo-type model \
        --local-dir "$CHECKPOINT_DIR/moge" \
        Ruicheng/moge-vitl
    
    if [ -f "$CHECKPOINT_DIR/moge/model.pt" ] || [ -f "$CHECKPOINT_DIR/moge/model.safetensors" ]; then
        echo "✓ MoGe checkpoints downloaded"
    else
        echo "✗ Error: MoGe checkpoints not found in download"
        echo "  Expected: model.pt or model.safetensors"
        echo "  Found files:"
        ls -la "$CHECKPOINT_DIR/moge/" 2>/dev/null | head -10
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "Checkpoint Summary"
echo "=============================================="
echo ""

# Show checkpoint sizes
echo "SAM 3D Objects:"
if [ -d "$CHECKPOINT_DIR/sam3d/checkpoints" ]; then
    du -sh "$CHECKPOINT_DIR/sam3d/checkpoints" | cut -f1 | xargs -I {} echo "  Size: {}"
    ls -la "$CHECKPOINT_DIR/sam3d/checkpoints/" 2>/dev/null | head -5
else
    echo "  Not downloaded"
fi

echo ""
echo "MoGe:"
if [ -f "$CHECKPOINT_DIR/moge/model.pt" ] || [ -f "$CHECKPOINT_DIR/moge/model.safetensors" ]; then
    du -sh "$CHECKPOINT_DIR/moge" | cut -f1 | xargs -I {} echo "  Size: {}"
    ls -la "$CHECKPOINT_DIR/moge/" 2>/dev/null | head -5
else
    echo "  Not downloaded"
fi

echo ""
echo "=============================================="
echo "Initialization complete!"
echo "=============================================="


# MARS Quick Start Guide

MARS (Multi Asset Reconstruction for Simulation) is a pipeline for generating Isaac Sim-ready 3D scenes from 2D images. It detects objects, segments them, reconstructs 3D meshes, and composes a complete scene with physics properties exported in USD format.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Building the Container](#building-the-container)
4. [Running the Pipeline](#running-the-pipeline)
5. [Pipeline Stages](#pipeline-stages)
6. [Configuration](#configuration)
7. [Output Structure](#output-structure)
8. [Using with Isaac Sim](#using-with-isaac-sim)
9. [Troubleshooting](#troubleshooting)
10. [External References](#external-references)

---

## System Requirements

### Hardware

- NVIDIA GPU with CUDA support (tested on RTX 5090)
- Minimum 16GB GPU memory recommended for full pipeline
- 32GB system RAM recommended

### Software

- Docker with NVIDIA Container Toolkit
  - Installation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Docker Compose v2
- NVIDIA Driver 535+ (for CUDA 12.1 support)

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url> mars
cd mars
```

### 2. Create Environment File

Create a `.env` file in the project root with your tokens:

```bash
# Required for downloading models
HF_TOKEN=your_huggingface_token
HUGGING_FACE_HUB_TOKEN=your_huggingface_token

# Optional
GITHUB_TOKEN=your_github_token
```

Get a HuggingFace token at: https://huggingface.co/settings/tokens

### 3. Initialize Checkpoints Directory

Model checkpoints are stored on the host filesystem to persist across container rebuilds:

```bash
mkdir -p checkpoints/sam3d checkpoints/moge
```

---

## Building the Container

### Standard Build

```bash
cd develop
./scripts/build.sh
```

Build time: 15-30 minutes depending on network speed and cache.

### Low-Memory Build

For systems with limited memory during build:

```bash
./scripts/build-low-memory.sh
```

### Start the Container

```bash
./scripts/start.sh
```

### Enter the Container

```bash
./scripts/enter.sh
```

Inside the container, the Python virtual environment at `/opt/mars-env/` is automatically activated. The workspace is mounted at `/workspace/`.

---

## Running the Pipeline

### Basic Usage

```bash
python run_pipeline.py <image_path>
```

### Command Line Options

```
python run_pipeline.py <image_path> [options]

Arguments:
  image_path              Path to input image

Options:
  -o, --output-dir DIR    Output directory (default: output)
  --run-until STAGE       Stop after this stage (default: composition)
  --enable-physics        Enable physics estimation
  --enable-validation     Enable PyBullet physics validation
  --enable-storage        Enable storage/indexing
  -v, --verbose           Debug logging
```

### Examples

Run full pipeline (ingestion through composition):

```bash
python run_pipeline.py /workspace/tests/data/test_image.jpg
```

Run only through detection:

```bash
python run_pipeline.py scene.jpg --run-until detection
```

Enable all optional stages:

```bash
python run_pipeline.py scene.jpg --enable-physics --enable-validation
```

Custom output location:

```bash
python run_pipeline.py scene.jpg -o /workspace/visualizations/my_scene
```

### Using Python API

```python
from src.mars import run

result = run(
    image_path="/workspace/tests/data/test_image.jpg",
    output_dir="output",
    run_until="composition",
    enable_physics=True,
    enable_validation=True
)

print(f"Status: {result['status']}")
print(f"Objects: {len(result['reconstruction']['objects'])}")
print(f"Scene: {result['composition']['scene_path']}")
```

---

## Testing

The MARS pipeline includes a test script that simplifies running the pipeline on test images. The script handles container execution, path mapping, and result visualization.

### Using the Test Script

The test script is located at `tests/scripts/test_pipeline.sh` and can be run from the project root:

```bash
./tests/scripts/test_pipeline.sh [image_path] [--run-until STAGE] [--output-dir DIR]
```

### Basic Usage

Run on default test image with full pipeline:

```bash
./tests/scripts/test_pipeline.sh
```

This uses the default test image at `tests/data/test_image_1.jpg` and runs the complete pipeline through the composition stage.

### Specifying a Test Image

Run on a specific image:

```bash
./tests/scripts/test_pipeline.sh tests/data/my_test_image.jpg
```

The script automatically maps paths:
- Images in `tests/` directory are accessible at `/workspace/tests/` in the container
- Project root files are accessible at `/workspace/` in the container

### Running Partial Pipeline

Test only specific stages:

```bash
# Run only through detection
./tests/scripts/test_pipeline.sh --run-until detection

# Run through segmentation
./tests/scripts/test_pipeline.sh --run-until segmentation

# Run through reconstruction
./tests/scripts/test_pipeline.sh --run-until reconstruction
```

Available stages:
- `ingestion` - Image validation only
- `detection` - Object detection
- `segmentation` - Mask generation
- `reconstruction` - 3D mesh creation
- `composition` - Full scene assembly (default)

### Custom Output Directory

Specify where results should be saved:

```bash
./tests/scripts/test_pipeline.sh --output-dir /workspace/visualizations/my_test
```

**Important**: To view results on the host machine, use paths under `/workspace/visualizations/`:
- Container path: `/workspace/visualizations/test_run/`
- Host path: `visualizations/test_run/` (relative to project root)

### Complete Example

Run a full test with custom image and output:

```bash
./tests/scripts/test_pipeline.sh \
    tests/data/kitchen_scene.jpg \
    --run-until composition \
    --output-dir /workspace/visualizations/kitchen_test
```

### What the Script Does

1. **Container Check**: Verifies the MARS container is running, starts it if needed
2. **Path Validation**: Checks that the test image exists
3. **Path Mapping**: Converts host paths to container paths automatically
4. **Pipeline Execution**: Runs the pipeline inside the container with proper Python path setup
5. **Result Summary**: Displays a summary of detected objects, masks, and generated meshes
6. **Output Location**: Shows where results are saved (both container and host paths)

### Viewing Results

After running the test script, results are organized by stage:

```
visualizations/test_run/
  <image_id>/
    1_ingestion/
      source.jpg
      metadata.json
    2_detection/
      detections.json
      visualization.jpg
    3_segmentation/
      masks/
        mask_0.png
        ...
      visualization.jpg
    4_reconstruction/
      objects/
        object_0/
          visual.obj
          collision.obj
          visual.usdc
        ...
    5_composition/
      scene.json
      scene.usdc
```

### Troubleshooting Test Script

**Container not running**: The script will attempt to start it automatically. If it fails:

```bash
cd develop
./scripts/start.sh
```

**Image not found**: Check available test images:

```bash
find tests -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \)
```

**Results not visible on host**: Ensure you're using `/workspace/visualizations/` as the output directory prefix. The script maps this to the host `visualizations/` directory.

**Permission errors**: Make sure the script is executable:

```bash
chmod +x tests/scripts/test_pipeline.sh
```

### Running Unit Tests

For code-level unit tests, use pytest inside the container:

```bash
# Enter container
cd develop
./scripts/enter.sh

# Run tests
cd /workspace
python -m pytest tests/ -v
```

---

## Pipeline Stages

The pipeline consists of 8 stages. By default, stages 1-6 run (Phase 1). Stages 7-8 are optional.

### Stage 1: Ingestion

Validates and prepares input images.

- Checks resolution (minimum 512x512)
- Validates format (JPEG, PNG)
- Creates staging copy

### Stage 2: Detection

Identifies objects using vision-language models.

- Primary model: Qwen 2.5 VL (labels and size estimation)
- Secondary model: GroundingDINO (accurate bounding boxes)
- Outputs: Object labels, bounding boxes, estimated sizes, materials

Configuration: `config/detection.yaml`

### Stage 3: Segmentation

Generates precise masks using SAM 3.

- Input: Detection bounding boxes and labels
- Output: Per-object binary masks
- Deduplication of overlapping masks

Configuration: `config/segmentation.yaml`

### Stage 4: Reconstruction

Creates 3D meshes from 2D masks using SAM 3D Objects.

- Generates visual mesh (OBJ with vertex colors)
- Generates collision mesh (convex hull)
- Exports USDC format for each object
- Applies mesh repair (hole filling, normal fixing)

Configuration: `config/reconstruction.yaml`

### Stage 5: Physics (Optional)

Estimates physical properties.

- Mass calculation from volume and material density
- Friction and restitution coefficients
- Inertia tensor computation
- Center of mass estimation

Configuration: `config/physics.yaml`

### Stage 6: Composition

Arranges objects in 3D space.

- Depth estimation using Depth Anything V2
- Real-world scaling from category database or LLM estimation
- Stacking and support relationship detection
- Scene graph construction
- USD scene export (USDC format)

Configuration: `config/composition.yaml`

### Stage 7: Validation (Optional)

Physics simulation validation using PyBullet.

- Drop test: Objects placed and simulated
- Stability check: Measures final displacement
- Penetration detection: Checks for collisions

Configuration: `config/validation.yaml`

### Stage 8: Storage (Optional)

Scene indexing and export.

- Scene metadata storage
- Asset cataloging
- Export to various formats

Configuration: `config/storage.yaml`

---

## Configuration

All configuration files are in the `config/` directory and use YAML format.

### Detection Configuration (`config/detection.yaml`)

Key settings:

```yaml
detection:
  model:
    type: "hybrid"                              # hybrid or qwen_direct
    huggingface_model: "Qwen/Qwen2.5-VL-3B-Instruct"
  processing:
    box_threshold: 0.35                         # Detection confidence
    max_image_dimension: 1280                   # Resize large images
  estimate_sizes: true                          # LLM size estimation
```

### Composition Configuration (`config/composition.yaml`)

Key settings:

```yaml
composition:
  layout:
    scene_width: 1.0      # Scene bounds in meters
    scene_depth: 0.8
  depth:
    enabled: true
    model: "depth_anything_v2"
    model_size: "large"
  scaling:
    enabled: true
    default_scale: 0.1    # Default 10cm if unknown
    categories:
      laptop: 0.35
      cup: 0.1
      phone: 0.15
      # ... extensive category list
```

### Adding Custom Object Scales

Edit `config/composition.yaml` and add to the `categories` section:

```yaml
categories:
  my_custom_object: 0.2   # 20cm
```

Or add a pattern for substring matching:

```yaml
patterns:
  - pattern: "custom"
    scale: 0.2
```

---

## Output Structure

After running the pipeline, outputs are organized as follows:

```
output/
  <image_id>/
    1_ingestion/
      source.jpg           # Validated input image
      metadata.json        # Image metadata
    2_detection/
      detections.json      # Detected objects with bboxes
      visualization.jpg    # Detection overlay
    3_segmentation/
      masks/
        mask_0.png         # Binary mask
        mask_1.png
        ...
      visualization.jpg    # Mask overlay
    4_reconstruction/
      objects/
        laptop_mask_0/
          visual.obj       # Textured mesh
          collision.obj    # Physics collision mesh
          visual.usdc      # USD format
        phone_mask_1/
          ...
    5_composition/
      scene.json           # Scene graph
      scene.usdc           # Isaac Sim-ready USD scene
    6_validation/          # If enabled
      report.json
```

### Output Formats

- **OBJ**: Standard mesh format with vertex colors
- **USDC**: USD binary format for Isaac Sim/Omniverse
- **JSON**: Scene graph with transforms and physics properties

---

## Using with Isaac Sim

### Loading the Scene

The `scene.usdc` file can be loaded directly in Isaac Sim:

1. Open Isaac Sim
2. File > Open > Select `scene.usdc`
3. Objects appear with:
   - Correct positions and scales
   - Collision meshes attached
   - Physics properties (mass, friction, restitution)

### Scene Structure

The USD scene follows Isaac Sim conventions:

```
/World
  /objects
    /laptop_mask_0
      - RigidBodyAPI
      - MassAPI
      - CollisionAPI
    /phone_mask_1
      ...
```

### Physics Properties

Each object includes:

- Mass (kg)
- Static and dynamic friction
- Restitution (bounciness)
- Center of mass

---

## Troubleshooting

### CUDA Not Available

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Ensure the container was started with `--gpus all`:

```bash
docker inspect mars-container | grep -A5 DeviceRequests
```

### Out of Memory (OOM)

For large images, reduce `max_image_dimension` in `config/detection.yaml`:

```yaml
processing:
  max_image_dimension: 1024  # Reduce from 1280
```

Or use a smaller model:

```yaml
model:
  huggingface_model: "Qwen/Qwen2-VL-2B-Instruct"  # Smaller than 3B
```

### Objects Not Detected

Lower the detection threshold in `config/detection.yaml`:

```yaml
processing:
  box_threshold: 0.25  # Lower from 0.35
```

### Incorrect Object Sizes

Add the object to `config/composition.yaml`:

```yaml
categories:
  my_object: 0.15  # Size in meters
```

Or enable LLM size estimation:

```yaml
estimate_sizes: true
```

### Mesh Quality Issues

The pipeline applies mesh repair automatically. For persistent issues:

1. Check the source image quality
2. Ensure the object is clearly visible
3. Consider using higher resolution input

### Prefect Errors

The pipeline runs in ephemeral mode by default. If you see "stopped service" errors, they are suppressed and do not affect pipeline execution.

To run with a Prefect server:

```bash
# Start Prefect server
prefect server start

# Set environment variable
export PREFECT_API_URL=http://127.0.0.1:4200/api
```

### Container Build Fails

For memory issues during build:

```bash
./scripts/build-low-memory.sh
```

For network issues, ensure HuggingFace token is set:

```bash
echo $HF_TOKEN  # Should show your token
```

---

## External References

### Models Used

- **Qwen 2.5 VL**: Vision-Language model for object detection and understanding
  - https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
  
- **GroundingDINO**: Text-conditioned object detection
  - https://github.com/IDEA-Research/GroundingDINO
  - https://huggingface.co/IDEA-Research/grounding-dino-tiny

- **SAM 3 (Segment Anything Model)**: Image segmentation
  - https://github.com/facebookresearch/segment-anything

- **SAM 3D Objects**: Single-view 3D reconstruction
  - https://github.com/facebookresearch/sam3d

- **Depth Anything V2**: Monocular depth estimation
  - https://github.com/DepthAnything/Depth-Anything-V2
  - https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf

### Frameworks

- **Prefect**: Workflow orchestration
  - https://docs.prefect.io/

- **PyBullet**: Physics simulation for validation
  - https://pybullet.org/

- **USD (Universal Scene Description)**: Scene format
  - https://openusd.org/
  - https://graphics.pixar.com/usd/docs/

- **Isaac Sim**: Robot simulation
  - https://developer.nvidia.com/isaac-sim

### Libraries

- **Trimesh**: 3D mesh processing
  - https://trimsh.org/

- **Open3D**: 3D data processing
  - http://www.open3d.org/

- **Transformers**: HuggingFace model loading
  - https://huggingface.co/docs/transformers/

---

## Support

For issues:

1. Check container logs: `./scripts/logs.sh`
2. Run with verbose logging: `python run_pipeline.py image.jpg -v`
3. Verify dependencies: `pip list | grep -E "torch|trimesh|transformers"`


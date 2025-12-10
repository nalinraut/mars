# MARS - Multi Asset Reconstruction for Simulation

Transform 2D images into physics-ready 3D scene assets for robotics training.

## Overview

MARS (Multi Asset Reconstruction for Simulation) is a complete pipeline that:
- **Segments** objects from images using SAM (Segment Anything Model)
- **Reconstructs** full 3D geometry and textures using SAM 3D Objects
- **Estimates** physics properties (mass, friction, inertia)
- **Validates** scenes with MuJoCo physics simulation
- **Exports** to multiple formats (USD, MJCF, URDF)

## Architecture

```
Image → SAM Segmentation → SAM 3D Objects → Physics → MuJoCo Validation → Export
```

### Pipeline Stages

1. **Ingestion** - Image validation and deduplication
2. **Segmentation** - Object detection with SAM
3. **3D Reconstruction** - Mesh generation with SAM 3D Objects
4. **Physics Estimation** - Material properties and mass calculation
5. **Scene Composition** - Spatial layout reconstruction
6. **Validation** - MuJoCo physics simulation testing
7. **Storage** - Indexing and export to multiple formats
   
See `docs/DESIGN.md` for detailed setup and usage instructions.

## Quick Start

```bash
# Build and start container
cd develop
./scripts/build.sh
./scripts/start.sh

# Run test pipeline
./tests/scripts/test_pipeline.sh
```

See `docs/QUICKSTART.md` for detailed setup and usage instructions.

## Development

### Project Structure

```
MARS/
├── develop/              # Docker and development tools
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── scripts/
│       ├── build.sh
│       ├── start.sh
│       ├── enter.sh
│       ├── stop.sh
│       └── logs.sh
├── src/                  # Source code (mounted to container)
│   └── mars/
│       ├── pipeline.py   # Main pipeline
│       ├── ingestion/    # Stage 1
│       ├── segmentation/ # Stage 2
│       ├── reconstruction/ # Stage 3
│       ├── physics/      # Stage 4
│       ├── composition/  # Stage 5
│       ├── validation/   # Stage 6
│       ├── storage/      # Stage 7
│       ├── exporters/    # Export utilities
│       └── utils/        # Shared utilities
├── config/               # Configuration files
├── tests/                # Test suite
└── README.md             # This file
```

### Source Code is Mounted

The `src/` directory is mounted to `/workspace/src` in the container, so:
- Edit code on your host machine
- Changes are immediately available in the container
- No need to rebuild for code changes

### Development Workflow

```bash
# 1. Edit code on host
vim src/mars/segmentation/segmentation.py

# 2. Test in container
cd develop
./scripts/enter.sh

# Inside container:
python -m pytest /workspace/src/tests/

# 3. Run pipeline
python -m mars.pipeline --input ...
```

## Container Management

```bash
cd develop

# Build container
./scripts/build.sh

# Start container
./scripts/start.sh

# Enter container
./scripts/enter.sh

# View logs
./scripts/logs.sh

# Stop container
./scripts/stop.sh

# Rebuild from scratch
./scripts/rebuild.sh
```

## Configuration

Edit `config/pipeline.yaml` to customize:
- Model checkpoints
- Quality thresholds
- Physics parameters
- Export formats

## Testing

Use the test script to run the pipeline on test images:

```bash
./tests/scripts/test_pipeline.sh [image_path] [--run-until STAGE] [--output-dir DIR]
```

See `docs/QUICKSTART.md` for detailed testing instructions.

## Integration 

**Output Contract:**
- Scene configs (JSON with meshes, physics, zones)
- Queryable index (task type, objects, quality)
- Multiple export formats (USD, MJCF, URDF)
- Randomization bounds for domain randomization

## Documentation

- `develop/docs/` - Full architecture documentation
- `config/` - Configuration examples
- `src/mars/` - Code documentation (docstrings)

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

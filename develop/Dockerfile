# MARS Container - Multi Asset Reconstruction for Simulation
# Python 3.11 with PyBullet, SAM, and Computer Vision Stack
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Limit build parallelism to reduce memory usage during compilation
# This prevents OOM errors when building memory-intensive packages
ENV MAX_JOBS=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV NUMEXPR_MAX_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libglfw3 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    build-essential \
    cmake \
    ninja-build \
    libglm-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Configure git to avoid credential prompts during Docker builds
# This prevents git clone from hanging on authentication prompts for public repos
ENV GIT_TERMINAL_PROMPT=0
ENV GIT_ASKPASS=echo
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" && \
    git config --global --unset-all credential.helper 2>/dev/null || true && \
    git config --global http.sslVerify true

# Create virtual environment for MARS
RUN python3.11 -m venv /opt/mars-env

# Upgrade pip in the virtual environment
RUN /opt/mars-env/bin/pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY develop/requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA support (compatible with SAM 3D Objects and xFormers)
# PyTorch must be installed separately due to custom index URL
# Install PyTorch 2.9.0+cu128 for RTX 5090 support (CUDA 12.8 with sm_120 architecture)
# PyTorch 2.9.0+ is required for xFormers compatibility and RTX 5090 (Blackwell) support
RUN /opt/mars-env/bin/pip install --no-cache-dir \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install xFormers (Required for MoGe and modern SAM models)
# Must be installed after PyTorch 2.9.0+ for compatibility
# Limit build parallelism to reduce memory usage
RUN MAX_JOBS=1 /opt/mars-env/bin/pip install --no-cache-dir xformers || \
    (export MAX_JOBS=1 && /opt/mars-env/bin/pip install --no-cache-dir xformers)

# Install Flash Attention (Optional - speeds up some SAM models)
# NOTE: Building from source because PyTorch 2.9.0+cu128 (CUDA 12.8) is too new
#       - No pre-built wheels available for this PyTorch/CUDA combination
#       - Pre-built wheels exist for PyTorch 2.0-2.4 with CUDA 11.8/12.1
#       - Building from source takes 1-2 hours but ensures compatibility
#       - If build fails/times out, pipeline will work without it (just slower)
RUN /opt/mars-env/bin/pip install --no-cache-dir packaging psutil && \
    (echo "Attempting to install flash-attn (will build from source - no pre-built wheels for PyTorch 2.9.0+cu128)..." && \
     MAX_JOBS=1 /opt/mars-env/bin/pip install --no-cache-dir flash-attn --no-build-isolation || \
     (export MAX_JOBS=1 && /opt/mars-env/bin/pip install --no-cache-dir flash-attn --no-build-isolation) || \
     (echo "WARNING: flash-attn build failed or timed out. Continuing without it (models will work but may be slower)." && true))

# Install SAM 3D Objects dependencies FIRST (before our requirements to avoid conflicts)
# Install critical packages individually to avoid git build issues
RUN echo "Installing SAM 3D Objects core dependencies..." && \
    # Install utils3d from GitHub source (will be cloned and built later)
    # Skip PyPI install - we'll build from source
    # Install gsplat from PyPI (works better than git version which has build issues)
    /opt/mars-env/bin/pip install --no-cache-dir gsplat && \
    # Install kaolin pre-built wheel (faster than building from source)
    # Use PyTorch 2.4.0 wheel (compatible with 2.5.1)
    # Note: Kaolin causes segfaults, so we'll create a stub instead
    echo "Installing kaolin pre-built wheel (will be replaced with stub due to segfaults)..." && \
    /opt/mars-env/bin/pip install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html || \
    (echo "Warning: kaolin installation failed - will create stub" && true) && \
    # Create kaolin stub to prevent segfaults (kaolin is only used for visualization)
    /opt/mars-env/bin/python3 << 'KAOLIN_STUB'
import os
import sys
kaolin_path = '/opt/mars-env/lib/python3.11/site-packages/kaolin'
stub_dir = kaolin_path + '_stub'
if os.path.exists(kaolin_path):
    # Backup real kaolin
    os.rename(kaolin_path, kaolin_path + '_disabled')
# Create stub modules
os.makedirs(stub_dir, exist_ok=True)
with open(os.path.join(stub_dir, '__init__.py'), 'w') as f:
    f.write('# Kaolin stub - prevents segfault\npass\n')
os.makedirs(os.path.join(stub_dir, 'visualize'), exist_ok=True)
with open(os.path.join(stub_dir, 'visualize', '__init__.py'), 'w') as f:
    f.write('IpyTurntableVisualizer = None\n')
os.makedirs(os.path.join(stub_dir, 'render', 'camera'), exist_ok=True)
with open(os.path.join(stub_dir, 'render', '__init__.py'), 'w') as f:
    f.write('# stub\n')
with open(os.path.join(stub_dir, 'render', 'camera', '__init__.py'), 'w') as f:
    f.write('Camera = CameraExtrinsics = PinholeIntrinsics = None\n')
os.makedirs(os.path.join(stub_dir, 'utils', 'testing'), exist_ok=True)
with open(os.path.join(stub_dir, 'utils', '__init__.py'), 'w') as f:
    f.write('# stub\n')
with open(os.path.join(stub_dir, 'utils', 'testing', '__init__.py'), 'w') as f:
    f.write('def check_tensor(*args, **kwargs): return True\n')
os.rename(stub_dir, kaolin_path)
print('Created kaolin stub to prevent segfaults')
KAOLIN_STUB

# Install utils3d from GitHub source (PyPI version doesn't have numpy submodule)
# Clone and build from source
RUN echo "Installing utils3d from GitHub source..." && \
    /opt/mars-env/bin/pip uninstall -y utils3d 2>/dev/null || true && \
    cd /tmp && \
    git clone https://github.com/EasternJournalist/utils3d.git && \
    /opt/mars-env/bin/pip install --no-cache-dir ./utils3d && \
    rm -rf /tmp/utils3d || true

# Install missing dependencies for SAM 3D Objects (from requirements.txt)
# Install critical ones individually to reduce memory pressure
# Note: Excluding pyvista to avoid vtk dependency (425MB, not used)
RUN /opt/mars-env/bin/pip install --no-cache-dir loguru timm trimesh optree astor opencv-python lightning spconv-cu121 xatlas pymeshfix python-igraph easydict plyfile imageio

# Install PyTorch3D from git (must be built from source, needs torch available)
# Use single-threaded build to reduce memory usage
# This is the most memory-intensive step - build with minimal parallelism
RUN echo "Installing PyTorch3D (this may take 15-20 minutes, using single-threaded build to reduce memory)..." && \
    MAX_JOBS=1 /opt/mars-env/bin/pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47" 2>&1 | tee /tmp/pytorch3d_build.log || \
    (echo "Warning: pytorch3d build failed with specific commit, trying latest..." && \
     MAX_JOBS=1 /opt/mars-env/bin/pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" 2>&1 | tee /tmp/pytorch3d_build_latest.log || \
     (echo "ERROR: pytorch3d installation failed. Check logs:" && cat /tmp/pytorch3d_build*.log 2>/dev/null || true))

# Install our requirements AFTER SAM 3D Objects (to allow version resolution)
# Install critical packages first (sam3, open3d) that SAM 3D Objects needs
RUN /opt/mars-env/bin/pip install --no-cache-dir sam3==0.1.2 open3d==0.18.0 && \
    /opt/mars-env/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Fix SAM 3 tokenizer vocabulary file location
    # SAM 3 expects bpe_simple_vocab_16e6.txt.gz in site-packages/assets/
    # but it's installed in open_clip. Create assets directory and copy it.
    echo "Setting up SAM 3 tokenizer vocabulary file..." && \
    mkdir -p /opt/mars-env/lib/python3.11/site-packages/assets && \
    if [ -f /opt/mars-env/lib/python3.11/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz ]; then \
        cp /opt/mars-env/lib/python3.11/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz \
           /opt/mars-env/lib/python3.11/site-packages/assets/ && \
        echo "✓ Copied SAM 3 tokenizer vocabulary file to assets/"; \
    else \
        echo "WARNING: open_clip tokenizer file not found. SAM 3 may fail at runtime."; \
    fi

# Install SAM 3D Objects - Meta's 3D reconstruction from single images
# Clone and install the repository (needed for model classes and inference code)
# Set up extra index URLs for NVIDIA packages
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# Note: utils3d is already installed above, so we skip duplicate installation
RUN cd /opt && \
    git clone https://github.com/facebookresearch/sam-3d-objects.git && \
    cd sam-3d-objects && \
    echo "Installing SAM 3D Objects dependencies..." && \
    # Install main requirements (skip problematic packages if needed)
    # Install one package at a time to reduce memory pressure
    /opt/mars-env/bin/pip install --no-cache-dir -r requirements.txt || \
    (echo "Warning: Some requirements.txt packages failed, continuing..." && true) && \
    # Install inference requirements (kaolin, gsplat, etc.) - skip already installed
    /opt/mars-env/bin/pip install --no-cache-dir -r requirements.inference.txt || \
    (echo "Warning: Some inference requirements failed, continuing..." && true) && \
    # Install PyTorch3D requirements (pytorch3d, flash_attn) - skip already installed
    /opt/mars-env/bin/pip install --no-cache-dir -r requirements.p3d.txt || \
    (echo "Warning: Some p3d requirements failed, continuing..." && true) && \
    # Install SAM 3D Objects package itself
    /opt/mars-env/bin/pip install --no-cache-dir -e . || \
    (echo "Warning: SAM 3D Objects package installation failed. May need manual setup." && true)

# Set up workspace
WORKDIR /workspace

# Checkpoints are downloaded at runtime to host-mounted directory
# This allows:
#   1. Faster Docker builds (no 2GB+ downloads during build)
#   2. Persistent storage across container rebuilds
#   3. Easy checkpoint management on host
# See: /workspace/scripts/init_checkpoints.sh

# Create directory structure
RUN mkdir -p /workspace/mars && \
    mkdir -p /workspace/data/staging && \
    mkdir -p /workspace/data/scenes && \
    mkdir -p /workspace/data/cache && \
    mkdir -p /workspace/models && \
    mkdir -p /workspace/config && \
    mkdir -p /workspace/logs

# Create MARS environment activation script
RUN echo '#!/bin/bash\n\
echo "Activating MARS environment..."\n\
source /opt/mars-env/bin/activate\n\
export MARS_ENV_PATH="/opt/mars-env"\n\
echo "MARS environment activated!"\n\
echo "   - Python: $(python --version)"\n\
echo "   - PyTorch: $(python -c \"import torch; print(torch.__version__)\" 2>/dev/null || echo \"Not loaded\")"\n\
echo "   - CUDA Available: $(python -c \"import torch; print(torch.cuda.is_available())\" 2>/dev/null || echo \"Not loaded\")"\n\
echo "   - PyBullet: $(python -c \"import pybullet; print(pybullet.__version__)\" 2>/dev/null || echo \"Not loaded\")"\n\
echo ""\n\
echo "Pipeline Stages:"\n\
echo "   1. Ingestion → 2. Segmentation → 3. Reconstruction"\n\
echo "   4. Physics → 5. Composition → 6. Validation → 7. Storage"\n\
echo ""\n\
echo "Available Tools:"\n\
echo "   - SAM 2 (Image/Video Segmentation)"\n\
echo "   - SAM 3D Objects (3D Reconstruction)"\n\
echo "   - PyBullet Physics Engine"\n\
echo "   - Trimesh/Open3D for 3D Processing"\n\
echo "   - Prefect for Orchestration"\n\
echo "   - Ray for Distributed Processing"\n\
echo ""\n\
if [ $# -eq 0 ]; then\n\
    exec /bin/bash\n\
else\n\
    exec "$@"\n\
fi' > /usr/local/bin/mars-env && \
chmod +x /usr/local/bin/mars-env

# Configure environment
ENV MARS_ENV_PATH=/opt/mars-env
ENV PATH=/opt/mars-env/bin:${PATH}

# Configure SAM 3D Objects - skip init module (it doesn't exist in the repository)
ENV LIDRA_SKIP_INIT=1
# Set CUDA_HOME directly (SAM 3D Objects tries to get it from CONDA_PREFIX, but we use venv)
ENV CUDA_HOME=/usr/local/cuda
# Set VIRTUAL_ENV for venv compatibility
ENV VIRTUAL_ENV=/opt/mars-env
# Set LD_LIBRARY_PATH to include PyTorch libraries (needed for kaolin and other compiled extensions)
ENV LD_LIBRARY_PATH=/opt/mars-env/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
# Patch SAM 3D Objects inference.py to work with venv instead of conda
# The original code does: os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
# We'll patch it to use CUDA_HOME if set, otherwise use VIRTUAL_ENV or default
# Copy patch scripts
COPY develop/scripts/patch_pointmap.py /tmp/patch_pointmap.py
COPY develop/scripts/patch_image_mesh.py /tmp/patch_image_mesh.py

# Also make moge optional in pointmap.py (moge is only needed for pointmap intrinsics inference)
RUN cd /opt/sam-3d-objects && \
    if [ -f notebook/inference.py ]; then \
        sed -i 's|os.environ\["CUDA_HOME"\] = os.environ\["CONDA_PREFIX"\]|os.environ["CUDA_HOME"] = os.environ.get("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/usr/local/cuda"))|g' notebook/inference.py && \
        echo "Patched inference.py to work with venv"; \
    fi && \
    if [ -f sam3d_objects/pipeline/utils/pointmap.py ]; then \
        python3 /tmp/patch_pointmap.py sam3d_objects/pipeline/utils/pointmap.py || echo "pointmap.py patch skipped"; \
    fi && \
    if [ -f sam3d_objects/utils/visualization/image_mesh.py ]; then \
        python3 /tmp/patch_image_mesh.py sam3d_objects/utils/visualization/image_mesh.py || echo "image_mesh.py patch skipped"; \
    fi

# Install MoGe (Monocular Geometry Estimation) from GitHub
# MoGe provides accurate depth estimation for 3D reconstruction
# Note: MoGe requires gradio, but we remove it immediately after install (not used at runtime)
RUN echo "Installing MoGe from GitHub..." && \
    /opt/mars-env/bin/pip install --no-cache-dir \
        "git+https://github.com/microsoft/MoGe.git" && \
    /opt/mars-env/bin/pip uninstall -y gradio 2>/dev/null || true && \
    echo "MoGe installed successfully (gradio removed - not used at runtime)"

# Copy checkpoint initialization script
COPY develop/scripts/init_checkpoints.sh /usr/local/bin/init_checkpoints.sh
RUN chmod +x /usr/local/bin/init_checkpoints.sh

# Copy Prefect server start script
COPY develop/scripts/start_prefect.sh /usr/local/bin/start_prefect.sh
RUN chmod +x /usr/local/bin/start_prefect.sh

# Remove build tools to reduce container size (no longer needed after package installation)
RUN apt-get remove -y \
    build-essential \
    cmake \
    ninja-build \
    python3.11-dev \
    libglm-dev \
    libeigen3-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache/pip

# Remove SAM1 and SAM2 directories (unused, replaced by SAM3)
RUN rm -rf /opt/sam2 2>/dev/null || true
RUN rm -rf /opt/mars-env/lib/python3.11/site-packages/segment_anything* 2>/dev/null || true

# Set up bashrc - auto-activate MARS environment
RUN echo "source /opt/mars-env/bin/activate" >> /root/.bashrc && \
    echo "echo 'MARS Container - Multi Asset Reconstruction for Simulation'" >> /root/.bashrc && \
    echo "echo 'Virtual environment activated: /opt/mars-env'" >> /root/.bashrc && \
    echo "echo 'Available: python, mars-env, pybullet'" >> /root/.bashrc && \
    echo "echo 'Workspace: /workspace/mars'" >> /root/.bashrc && \
    echo "echo ''" >> /root/.bashrc && \
    echo "# Initialize checkpoints on first run" >> /root/.bashrc && \
    echo "if [ ! -f /workspace/checkpoints/.initialized ]; then" >> /root/.bashrc && \
    echo "    echo 'First run - initializing checkpoints...'" >> /root/.bashrc && \
    echo "    /usr/local/bin/init_checkpoints.sh && touch /workspace/checkpoints/.initialized" >> /root/.bashrc && \
    echo "fi" >> /root/.bashrc

CMD ["/bin/bash"]


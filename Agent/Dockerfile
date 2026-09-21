# For CUDA/NVIDIA cloud gpu deployment image: JAX(CUDA 13)
# target platform linux/amd64 for NVIDIA GPU

# DEPLOYMENT COMMANDS:
# Build (on Apple silicon, cross-build for the deployment host):
#   docker buildx build --platform linux/amd64 -f Dockerfile.gpu -t ml-gpu .
# Run (on the GPU host, requires nvidia-container-toolkit):
#   docker run --rm --gpus all -v "$(pwd)":/workspace ml-gpu python (filename).py

FROM --platform=linux/amd64 python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        swig \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_gpu.text /tmp/requirements_gpu.text
RUN pip install -r /tmp/requirements_gpu.text && rm /tmp/requirements_gpu.text

RUN useradd -m -u 1000 user
USER user
WORKDIR /workspace

# Version Check
RUN python -c "import jax; \
    print('jax', jax.__version__);"

CMD ["bin/bash"]


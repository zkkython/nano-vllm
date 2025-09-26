#!/bin/bash
"""
Installation script for JAX dependencies

This script installs JAX and related dependencies for the JAX GPU implementation
of VocabParallelEmbedding.

Usage: bash scripts/install_jax.sh [--gpu] [--tpu]
"""

set -e

# Default options
INSTALL_GPU=false
INSTALL_TPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --tpu)
            INSTALL_TPU=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--gpu] [--tpu]"
            echo "  --gpu: Install JAX with GPU support (CUDA)"
            echo "  --tpu: Install JAX with TPU support"
            echo "  -h, --help: Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Installing JAX dependencies..."

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Install base JAX
echo "Installing base JAX..."
pip install --upgrade pip
pip install jax>=0.4.0

# Install JAX with GPU support if requested
if [ "$INSTALL_GPU" = true ]; then
    echo "Installing JAX with GPU support..."
    
    # Check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
        echo "Detected CUDA version: $cuda_version"
        
        # Install appropriate JAX version
        if [[ $(echo "$cuda_version >= 12.0" | bc -l) -eq 1 ]]; then
            pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        elif [[ $(echo "$cuda_version >= 11.8" | bc -l) -eq 1 ]]; then
            pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        else
            echo "Warning: CUDA version $cuda_version may not be supported"
            pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        fi
    else
        echo "Warning: nvidia-smi not found, installing generic CUDA support"
        pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    fi
fi

# Install JAX with TPU support if requested
if [ "$INSTALL_TPU" = true ]; then
    echo "Installing JAX with TPU support..."
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

# Install additional dependencies
echo "Installing additional dependencies..."
pip install flax>=0.8.0
pip install numpy>=1.21.0

# Verify installation
echo "Verifying installation..."
python3 -c "
import jax
import jax.numpy as jnp
import flax.linen as nn
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')

# Test basic functionality
rng = jax.random.PRNGKey(42)
x = jnp.array([1, 2, 3])
y = jnp.sin(x)
print(f'Basic JAX test passed: {y}')

print('Installation verification completed successfully!')
"

echo ""
echo "JAX installation completed!"
echo ""
echo "To test the VocabParallelEmbedding implementation:"
echo "  python examples/jax_embed_head_demo.py"
echo ""
echo "To run the complete demo:"
echo "  python examples/jax_complete_demo.py"
echo ""
echo "To compare with PyTorch:"
echo "  python examples/pytorch_jax_comparison.py"

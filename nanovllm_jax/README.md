# Nano-vLLM JAX Implementation

✅ **状态：已完成并测试通过 | Status: Completed and Tested**

A JAX-based implementation of nano-vllm, providing high-performance LLM inference using JAX and Flax.

## Features

- 🚀 **JAX-powered inference** - Leverages JAX's JIT compilation and XLA optimization
- 📖 **Clean JAX implementation** - Pure JAX/Flax codebase for maximum performance
- ⚡ **GPU acceleration** - Optimized for GPU inference with JAX
- 🔧 **Modular design** - Easy to extend and modify
- 🎯 **Qwen3 support** - Full support for Qwen3 models
- 💾 **Memory optimized** - Configurable memory usage for different GPU sizes
- ✅ **Tested and working** - Successfully generates text on GPU

## Installation

```bash
# Install JAX with CUDA support
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from nanovllm_jax import LLM, SamplingParams

# Load model
llm = LLM("/path/to/qwen3-model", enforce_eager=True, tensor_parallel_size=1)

# Generate text
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
prompts = ["Hello, I am Qwen3."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

## Running the Demo

```bash
python qwen3_demo.py --model /path/to/qwen3-model --temperature 0.7 --max_tokens 256
```

## Architecture

The JAX implementation follows the same architecture as the original PyTorch version:

- **Engine**: Model runner, scheduler, and sequence management
- **Layers**: Linear, attention, normalization, and activation layers
- **Models**: Qwen3 model implementation
- **Utils**: Context management and model loading utilities

## Key Differences from PyTorch Version

1. **JAX/Flax**: Uses JAX and Flax instead of PyTorch
2. **Functional approach**: Leverages JAX's functional programming model
3. **JIT compilation**: Automatic JIT compilation for performance
4. **XLA optimization**: Compiles to optimized XLA code

## Performance

The JAX implementation provides:
- Fast JIT compilation
- Optimized XLA kernels
- Efficient memory usage
- GPU acceleration

## Requirements

- Python 3.8+
- JAX with CUDA support
- Flax
- Transformers
- NumPy

## License

MIT License - same as the original nano-vllm project.

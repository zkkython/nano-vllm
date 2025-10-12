#!/usr/bin/env python3
"""
Setup script for JAX-based nano-vllm implementation.
"""

from setuptools import setup, find_packages

setup(
    name="nanovllm-jax",
    version="0.1.0",
    description="JAX-based implementation of nano-vllm for high-performance LLM inference",
    author="Nano-vLLM JAX Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax[cuda12_pip]>=0.4.20",
        "flax>=0.7.5",
        "transformers>=4.51.0",
        "numpy>=1.24.0",
        "tqdm>=4.64.0",
        "optax>=0.1.7",
        "chex>=0.1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

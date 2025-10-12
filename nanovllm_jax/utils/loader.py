import os
from typing import Dict, Any
import jax.numpy as jnp
from flax import linen as nn
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from safetensors import safe_open


def load_model_weights_from_safetensors(model_path: str) -> Dict[str, Any]:
    """Load model weights from safetensors format (JAX-friendly)."""
    print(f"Loading weights from {model_path}...")
    
    try:
        # Look for safetensors files
        safetensors_files = []
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                safetensors_files.append(os.path.join(model_path, file))
        
        if not safetensors_files:
            print("No safetensors files found, using random initialization...")
            return {}
        
        # Load the first safetensors file
        safetensors_file = safetensors_files[0]
        print(f"Loading from {safetensors_file}...")
        
        jax_params = {}
        
        with safe_open(safetensors_file, framework="numpy") as f:
            for key in f.keys():
                # Load tensor as numpy array
                tensor = f.get_tensor(key)
                
                # Convert to JAX array
                jax_tensor = jnp.array(tensor)
                
                # Handle nested structure
                keys = key.split('.')
                current = jax_params
                
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                current[keys[-1]] = jax_tensor
        
        print("✓ Model weights loaded successfully from safetensors")
        return jax_params
        
    except Exception as e:
        print(f"✗ Failed to load safetensors weights: {e}")
        return {}


def load_model_weights_from_numpy(model_path: str) -> Dict[str, Any]:
    """Load model weights from numpy format (if available)."""
    print(f"Trying to load numpy weights from {model_path}...")
    
    try:
        # Look for numpy weight files
        weight_files = []
        for file in os.listdir(model_path):
            if file.endswith('.npy') or file.endswith('.npz'):
                weight_files.append(os.path.join(model_path, file))
        
        if not weight_files:
            return {}
        
        # This would need to be implemented based on the specific format
        # For now, return empty
        print("Numpy weight loading not implemented yet...")
        return {}
        
    except Exception as e:
        print(f"✗ Failed to load numpy weights: {e}")
        return {}


def load_model_weights(model_path: str) -> Dict[str, Any]:
    """Load model weights from various formats."""
    # Try safetensors first (most common in modern HuggingFace models)
    weights = load_model_weights_from_safetensors(model_path)
    
    if weights:
        return weights
    
    # Try numpy format as fallback
    weights = load_model_weights_from_numpy(model_path)
    
    if weights:
        return weights
    
    # If all else fails, return empty dict (will use random initialization)
    print("No compatible weight format found, using random initialization...")
    return {}


def load_model(model: nn.Module, model_path: str) -> Dict[str, Any]:
    """Load model weights from HuggingFace format."""
    return load_model_weights(model_path)


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer from model path."""
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)

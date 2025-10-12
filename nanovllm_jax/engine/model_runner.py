import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Optional, Tuple
from flax import linen as nn
from flax.training import train_state

from nanovllm_jax.config import Config
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM
from nanovllm_jax.utils.context import set_context, get_context, reset_context
from nanovllm_jax.utils.loader import load_model


class ModelRunner:
    """JAX model runner for inference."""
    
    def __init__(self, config: Config, rank: int = 0):
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        
        # Initialize JAX
        self.devices = jax.devices()
        if len(self.devices) > 1:
            self.device = self.devices[rank % len(self.devices)]
        else:
            self.device = self.devices[0]
        
        # Initialize model
        self.model = Qwen3ForCausalLM(
            config=config.hf_config,
            tp_size=self.world_size,
            tp_rank=self.rank
        )
        
        # Initialize parameters
        self.params = self._initialize_params()
        
        # Initialize sampler
        self.sampler = Sampler()
        
        # Initialize KV cache
        self.kv_cache = self._allocate_kv_cache()
    
    def _initialize_params(self) -> Dict[str, Any]:
        """Initialize model parameters."""
        # Create dummy inputs for initialization with smaller size
        batch_size = 1
        seq_len = min(32, self.config.max_model_len)  # Use much smaller sequence length
        input_ids = jnp.zeros((batch_size,), dtype=jnp.int32)
        positions = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        # Initialize parameters with random values first
        params = self.model.init(
            jax.random.PRNGKey(0),
            input_ids,
            positions
        )
        
        # Try to load actual weights from HuggingFace model
        try:
            loaded_weights = load_model(self.model, self.config.model)
            if loaded_weights:
                print("Loading pre-trained weights...")
                # Merge loaded weights with initialized parameters
                params = self._merge_weights(params, loaded_weights)
                print("✓ Pre-trained weights loaded successfully")
            else:
                print("Using random initialization...")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Using random initialization...")
        
        # Handle weight tying if needed
        if self.config.hf_config.tie_word_embeddings:
            # Copy embedding weights to lm_head weights
            # Check the correct path: params['params']['model']['embed_tokens']
            if 'params' in params and 'model' in params['params'] and 'embed_tokens' in params['params']['model']:
                # Check for both 'embedding' and 'weight' keys
                if 'embedding' in params['params']['model']['embed_tokens']:
                    embed_weights = params['params']['model']['embed_tokens']['embedding']
                elif 'weight' in params['params']['model']['embed_tokens']:
                    embed_weights = params['params']['model']['embed_tokens']['weight']
                else:
                    print("Warning: No embedding weights found for weight tying")
                    embed_weights = None
                
                if embed_weights is not None and 'params' in params and 'lm_head' in params['params'] and 'weight' in params['params']['lm_head']:
                    params['params']['lm_head']['weight'] = embed_weights
        
        return params
    
    def _merge_weights(self, params: Dict[str, Any], loaded_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded weights with initialized parameters."""
        def merge_dict(d1, d2):
            """Recursively merge two dictionaries."""
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dict(d1[key], value)
                else:
                    d1[key] = value
            return d1
        
        # Create a copy of params to avoid modifying the original
        merged_params = jax.tree.map(lambda x: x, params)
        
        # The loaded weights have structure: {'model': {...}, 'lm_head': {...}}
        # The params have structure: {'params': {'model': {...}, 'lm_head': {...}}}
        # We need to map loaded_weights['model'] to merged_params['params']['model']
        # and loaded_weights['lm_head'] to merged_params['params']['lm_head']
        
        if 'model' in loaded_weights and 'params' in merged_params and 'model' in merged_params['params']:
            # Special handling for embedding weights: map 'weight' to 'embedding'
            if 'embed_tokens' in loaded_weights['model'] and 'embed_tokens' in merged_params['params']['model']:
                loaded_embed = loaded_weights['model']['embed_tokens']
                merged_embed = merged_params['params']['model']['embed_tokens']
                
                # Map 'weight' to 'embedding'
                if 'weight' in loaded_embed and 'embedding' in merged_embed:
                    merged_embed['embedding'] = loaded_embed['weight']
                
                # Copy other embed_tokens weights
                for key, value in loaded_embed.items():
                    if key != 'weight':  # Skip 'weight' as we already handled it
                        merged_embed[key] = value
            
            # Handle layer weights with special mapping
            if 'layers' in loaded_weights['model']:
                for layer_idx in range(self.config.hf_config.num_hidden_layers):
                    layer_key = f"Qwen3DecoderLayer_{layer_idx}"
                    if layer_key in merged_params['params']['model'] and str(layer_idx) in loaded_weights['model']['layers']:
                        layer_weights = loaded_weights['model']['layers'][str(layer_idx)]
                        layer_params = merged_params['params']['model'][layer_key]
                        
                        # Special handling for self_attn weights
                        if 'self_attn' in layer_weights and 'self_attn' in layer_params:
                            self_attn_weights = layer_weights['self_attn']
                            self_attn_params = layer_params['self_attn']
                            
                            # Map individual projections to QKV projection
                            if 'qkv_proj' in self_attn_params and all(k in self_attn_weights for k in ['q_proj', 'k_proj', 'v_proj']):
                                # Check if projections are arrays or dicts
                                q_proj = self_attn_weights['q_proj']
                                k_proj = self_attn_weights['k_proj'] 
                                v_proj = self_attn_weights['v_proj']
                                
                                # If they are dicts, extract the weight
                                if isinstance(q_proj, dict) and 'weight' in q_proj:
                                    q_proj = q_proj['weight']
                                if isinstance(k_proj, dict) and 'weight' in k_proj:
                                    k_proj = k_proj['weight']
                                if isinstance(v_proj, dict) and 'weight' in v_proj:
                                    v_proj = v_proj['weight']
                                
                                # Concatenate q, k, v projections
                                qkv_proj = jnp.concatenate([q_proj, k_proj, v_proj], axis=0)
                                self_attn_params['qkv_proj']['weight'] = qkv_proj
                            
                            # Copy other self_attn weights
                            for key, value in self_attn_weights.items():
                                if key not in ['q_proj', 'k_proj', 'v_proj']:  # Skip individual projections
                                    if key in self_attn_params:
                                        if isinstance(value, dict) and isinstance(self_attn_params[key], dict):
                                            merge_dict(self_attn_params[key], value)
                                        else:
                                            self_attn_params[key] = value
                        
                        # Merge other layer weights (mlp, layernorm, etc.)
                        for key, value in layer_weights.items():
                            if key != 'self_attn':  # Skip self_attn as we handled it above
                                if key in layer_params:
                                    merge_dict(layer_params[key], value)
                                else:
                                    layer_params[key] = value
            
            # Handle norm weights
            if 'norm' in loaded_weights['model'] and 'norm' in merged_params['params']['model']:
                merge_dict(merged_params['params']['model']['norm'], loaded_weights['model']['norm'])
        
        if 'lm_head' in loaded_weights and 'params' in merged_params and 'lm_head' in merged_params['params']:
            merge_dict(merged_params['params']['lm_head'], loaded_weights['lm_head'])
        
        return merged_params
    
    def _allocate_kv_cache(self) -> Dict[str, jnp.ndarray]:
        """Allocate KV cache."""
        hf_config = self.config.hf_config
        num_layers = hf_config.num_hidden_layers
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)
        block_size = self.config.kvcache_block_size
        num_blocks = self.config.num_kvcache_blocks
        
        if num_blocks <= 0:
            # Calculate number of blocks based on available memory - use smaller default
            num_blocks = 100  # Much smaller default value
        
        # Use smaller data type to save memory
        dtype = jnp.float16 if hf_config.torch_dtype == jnp.float32 else hf_config.torch_dtype
        
        kv_cache = {
            'k_cache': jnp.zeros((num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype),
            'v_cache': jnp.zeros((num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
        }
        
        return kv_cache
    
    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for prefill phase."""
        input_ids = []
        positions = []
        
        for seq in seqs:
            seq_len = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seq_len)))
        
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        positions = jnp.array(positions, dtype=jnp.int32)
        
        return input_ids, positions
    
    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for decode phase."""
        input_ids = []
        positions = []
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
        
        input_ids = jnp.array(input_ids, dtype=jnp.int32)
        positions = jnp.array(positions, dtype=jnp.int32)
        
        return input_ids, positions
    
    def prepare_sample(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare sampling parameters."""
        temperatures = [seq.temperature for seq in seqs]
        return jnp.array(temperatures, dtype=jnp.float32)
    
    def run_model(
        self, 
        input_ids: jnp.ndarray, 
        positions: jnp.ndarray, 
        is_prefill: bool
    ) -> jnp.ndarray:
        """Run the model forward pass."""
        # Set context for attention layers
        context = {
            'is_prefill': is_prefill,
            'kv_cache': self.kv_cache
        }
        set_context(context)
        
        # Forward pass with logits computation
        logits = self.model.apply(
            self.params,
            input_ids,
            positions,
            compute_logits=True
        )
        
        return logits
    
    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        """Run inference on sequences."""
        # Prepare inputs
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        
        # Run model
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # Sample tokens
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # Use current time as seed for randomness
        import time
        rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        token_ids = self.sampler(logits, temperatures, rng)
        
        # Reset context
        reset_context()
        
        return token_ids.tolist() if self.rank == 0 else []

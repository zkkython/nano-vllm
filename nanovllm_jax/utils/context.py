import threading
from typing import Optional, Dict, Any
import jax.numpy as jnp


# Thread-local storage for context
_context = threading.local()


def set_context(
    is_prefill: bool,
    cu_seqlens_q: Optional[jnp.ndarray] = None,
    cu_seqlens_k: Optional[jnp.ndarray] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    slot_mapping: Optional[jnp.ndarray] = None,
    context_lens: Optional[jnp.ndarray] = None,
    block_tables: Optional[jnp.ndarray] = None,
    kv_cache: Optional[Dict[str, jnp.ndarray]] = None
):
    """Set the current context for attention computation."""
    _context.is_prefill = is_prefill
    _context.cu_seqlens_q = cu_seqlens_q
    _context.cu_seqlens_k = cu_seqlens_k
    _context.max_seqlen_q = max_seqlen_q
    _context.max_seqlen_k = max_seqlen_k
    _context.slot_mapping = slot_mapping
    _context.context_lens = context_lens
    _context.block_tables = block_tables
    _context.kv_cache = kv_cache


def get_context() -> Dict[str, Any]:
    """Get the current context."""
    return {
        'is_prefill': getattr(_context, 'is_prefill', False),
        'cu_seqlens_q': getattr(_context, 'cu_seqlens_q', None),
        'cu_seqlens_k': getattr(_context, 'cu_seqlens_k', None),
        'max_seqlen_q': getattr(_context, 'max_seqlen_q', None),
        'max_seqlen_k': getattr(_context, 'max_seqlen_k', None),
        'slot_mapping': getattr(_context, 'slot_mapping', None),
        'context_lens': getattr(_context, 'context_lens', None),
        'block_tables': getattr(_context, 'block_tables', None),
        'kv_cache': getattr(_context, 'kv_cache', None)
    }


def reset_context():
    """Reset the current context."""
    _context.is_prefill = False
    _context.cu_seqlens_q = None
    _context.cu_seqlens_k = None
    _context.max_seqlen_q = None
    _context.max_seqlen_k = None
    _context.slot_mapping = None
    _context.context_lens = None
    _context.block_tables = None
    _context.kv_cache = None

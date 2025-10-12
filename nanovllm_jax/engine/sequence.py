import jax.numpy as jnp
from typing import List, Optional
from nanovllm_jax.sampling_params import SamplingParams


class Sequence:
    """Represents a sequence being processed."""
    
    def __init__(self, prompt: List[int], sampling_params: SamplingParams):
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.completion_token_ids: List[int] = []
        self.seq_id = id(self)
        self.is_finished = False
        self.num_cached_tokens = 0
        self.num_cached_blocks = 0
        self.num_blocks = 0
        self.last_block_num_tokens = 0
        self.block_table: List[int] = []
        self.last_token = prompt[-1] if prompt else 0
        
        # Initialize block table
        self._initialize_blocks()
    
    def _initialize_blocks(self):
        """Initialize block table for the sequence."""
        block_size = 256  # Default block size
        seq_len = len(self.prompt)
        self.num_blocks = (seq_len + block_size - 1) // block_size
        self.last_block_num_tokens = seq_len % block_size or block_size
        
        # Initialize block table with placeholder values
        self.block_table = list(range(self.num_blocks))
    
    def __len__(self) -> int:
        """Return the length of the sequence."""
        return len(self.prompt) + len(self.completion_token_ids)
    
    def __getitem__(self, index):
        """Get token at index or slice."""
        if isinstance(index, slice):
            # Handle slice indexing
            all_tokens = self.prompt + self.completion_token_ids
            return all_tokens[index]
        else:
            # Handle integer indexing
            if index < len(self.prompt):
                return self.prompt[index]
            else:
                return self.completion_token_ids[index - len(self.prompt)]
    
    def __iter__(self):
        """Iterate over all tokens in the sequence."""
        for token in self.prompt + self.completion_token_ids:
            yield token
    
    @property
    def temperature(self) -> float:
        """Get temperature for sampling."""
        return self.sampling_params.temperature
    
    def add_token(self, token: int):
        """Add a token to the completion."""
        self.completion_token_ids.append(token)
        self.last_token = token
        
        # Check if sequence is finished
        if token == -1 or len(self.completion_token_ids) >= self.sampling_params.max_tokens:
            self.is_finished = True

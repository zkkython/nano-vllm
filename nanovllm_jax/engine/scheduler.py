from typing import List, Tuple
from nanovllm_jax.engine.sequence import Sequence


class Scheduler:
    """Scheduler for managing sequences."""
    
    def __init__(self, config):
        self.config = config
        self.sequences: List[Sequence] = []
        self.finished_sequences: List[Sequence] = []
    
    def add(self, seq: Sequence):
        """Add a sequence to the scheduler."""
        self.sequences.append(seq)
    
    def schedule(self) -> Tuple[List[Sequence], bool]:
        """Schedule sequences for processing."""
        if not self.sequences:
            return [], False
        
        # Simple scheduling: process all sequences
        # In a full implementation, this would implement more sophisticated scheduling
        sequences_to_process = self.sequences.copy()
        is_prefill = any(seq.num_cached_tokens == 0 for seq in sequences_to_process)
        
        return sequences_to_process, is_prefill
    
    def postprocess(self, sequences: List[Sequence], token_ids: List[int]):
        """Post-process sequences after model execution."""
        for seq, token_id in zip(sequences, token_ids):
            seq.add_token(token_id)
            
            if seq.is_finished:
                self.finished_sequences.append(seq)
                self.sequences.remove(seq)
    
    def is_finished(self) -> bool:
        """Check if all sequences are finished."""
        return len(self.sequences) == 0

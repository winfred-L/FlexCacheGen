import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    model:                  Path to model directory
    max_num_batched_tokens: Maximum number of tokens in a batch / in Scheduler at once
    max_num_seqs:           Maximum number of concurrent sequences in Scheduler
    max_model_len:          Maximum sequence length
    gpu_memory_utilization: Fraction of GPU memory to use
    tensor_parallel_size:   Number of tensor parallel processes
    enforce_eager:          Whether to enforce eager execution / Disable CUDA graph optimization
    kvcache_block_size:     Size of each key-value cache block in tokens
    num_kvcache_blocks:     Total KV cache blocks (auto-calculated)
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

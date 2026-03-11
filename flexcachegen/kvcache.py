import torch
from flexcachegen.config import Config
# from transformers.cache_utils import DynamicLayer

class CacheLayer:
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    The max capacity is allocated after prefill and limited by `config.max_new_tokens`.
    New decoding KV is updated in flash-attn kernel `flash_attn_with_kvcache`, only `seq_len` needs manually controlled.
    """
    
    def __init__(self, max_new_tokens):
        self.seq_len = 0
        self.max_seq_len = 0
        self.max_new_tokens = max_new_tokens

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        device, dtype = key_states.device, key_states.dtype
        batch_size, seq_len, num_heads, head_dim = key_states.shape
        max_seq_len = seq_len + self.max_new_tokens

        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.keys = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.values = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.keys[:, :seq_len].copy_(key_states)
        self.values[:, :seq_len].copy_(value_states)

        
    

class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        self.gpu_buffer = [CacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)]
    
    def offload_layer_to_cpu(self, layer_idx):
        # TODO
        pass
    
    def load_layer_to_gpu(self, layer_idx):
        # do nothing now, only return specific layer's cache
        return self.gpu_buffer[layer_idx]

    def clear(self):
        self.gpu_buffer = [CacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)]



class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management



import torch
from flexcachegen.config import Config
# from transformers.cache_utils import DynamicLayer

class DynamicLayer:
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    
    ref: transformers.cache_utils.DynamicLayer
    """
    
    def __init__(self, max_new_tokens):
        # self.is_initialized = False
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
        # self.is_initialized = True
        

    # def update(
    #     self,
    #     key_states: torch.Tensor,
    #     value_states: torch.Tensor,
    # ):
    #     """
    #     Update the key and value caches in-place, and return the updated keys and value states.
    #     """
    #     # prefill stage
    #     if not self.is_initialized:
    #         self.lazy_initialization(key_states, value_states)
    #     # decoding stage
    #     else:
    #         assert self.seq_len < self.max_seq_len
    #         self.keys[:, self.seq_len : self.seq_len + 1].copy_(key_states)
    #         self.values[:, self.seq_len : self.seq_len + 1].copy_(value_states)
    #         self.seq_len += 1

    #     return self.keys[:, :self.seq_len], self.values[:, :self.seq_len]
        
    

class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        self.gpu_buffer = [DynamicLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)]
    
    def offload_layer_to_cpu(self, layer_idx):
        # TODO
        pass
    
    def load_layer_to_gpu(self, layer_idx):
        return self.gpu_buffer[layer_idx]

    def clear(self):
        self.gpu_buffer = [DynamicLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)]



class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management



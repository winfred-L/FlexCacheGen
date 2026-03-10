import torch
from flexcachegen.config import Config
# from transformers.cache_utils import DynamicLayer

class DynamicLayer:
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    
    ref: transformers.cache_utils.DynamicLayer
    """
    
    def __init__(self):
        self.is_initialized = False
        self.seq_len = 0
        self.max_seq_len = 0

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        device, dtype = key_states.device, key_states.dtype
        batch_size, seq_len, num_heads, head_dim = key_states.shape
        max_seq_len = 0
        while max_seq_len < seq_len:
            max_seq_len += 1024

        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.keys = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.keys[:, :seq_len].copy_(key_states)
        self.values = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.values[:, :seq_len].copy_(value_states)
        self.is_initialized = True
        

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """
        Update the key and value caches in-place, and return the updated keys and value states.
        """
        # prefill stage
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        # decoding stage
        else:
            self.keys[:, self.seq_len : self.seq_len + 1].copy_(key_states)
            self.values[:, self.seq_len : self.seq_len + 1].copy_(value_states)
            self.seq_len += 1

            if self.seq_len == self.max_seq_len:
                self.expand()

        return self.keys[:, :self.seq_len], self.values[:, :self.seq_len]
    
    
    def expand(self):
        """
        Expand the capacity of the key and value caches when the current 
        sequence length reaches the maximum allocated length.
        """
        # 1. Calculate new capacity
        old_max_seq_len = self.max_seq_len
        new_max_seq_len = old_max_seq_len * 2
        
        # 2. Preserve tensor properties (device, dtype, other dimensions)
        device = self.keys.device
        dtype = self.keys.dtype
        batch_size, _, num_heads, head_dim = self.keys.shape
        
        # 3. Allocate new larger tensors
        new_keys = torch.empty(
            (batch_size, new_max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        new_values = torch.empty(
            (batch_size, new_max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        
        # 4. Copy existing valid data to the new tensors
        new_keys[:, :old_max_seq_len, :, :].copy_(self.keys)
        new_values[:, :old_max_seq_len, :, :].copy_(self.values)
        
        # 5. Update internal state to point to new buffers
        self.keys = new_keys
        self.values = new_values
        self.max_seq_len = new_max_seq_len
        
    

class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        self.gpu_buffer = [DynamicLayer() for _ in range(self.num_hidden_layers)]
    
    def offload_layer_to_cpu(self, layer_idx):
        # TODO
        pass
    
    def load_layer_to_gpu(self, layer_idx):
        return self.gpu_buffer[layer_idx]

    def clear(self):
        self.gpu_buffer = [DynamicLayer() for _ in range(self.num_hidden_layers)]



class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management



import torch
from flexcachegen.config import Config


class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config

        batch = 1
        seq = 5000 # TODO
        num_kv_heads = config.hf_config.text_config.num_key_value_heads
        head_dim = config.hf_config.text_config.hidden_size // config.hf_config.text_config.num_attention_heads

        self.gpu_k_buffer = torch.empty(batch, seq, num_kv_heads, head_dim, device=config.device, dtype=torch.bfloat16)
        self.gpu_v_buffer = torch.empty(batch, seq, num_kv_heads, head_dim, device=config.device, dtype=torch.bfloat16)

        self.cache_seqlens = torch.zeros((batch,), device=config.device, dtype=torch.long)
    
    def offload_layer_to_cpu(self, layer_idx): pass
    def load_layer_to_gpu(self, layer_idx): pass

    def clear(self):
        pass



class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management



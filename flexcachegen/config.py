import os
import torch
from transformers import AutoConfig

# TODO: currently only support for qwen3vl-8b
MODEL_REGISTRY: dict[str, str] = {
    'qwen3vl-8b': '/data/lyc/models/Qwen3-VL-8B-Instruct',
}

class Config:
    model_type: str

    ##### model settings ##### (inferred from model_type when initialized)
    model_path: str
    hf_config: AutoConfig
    dtype: torch.dtype
    eos_token_id: tuple[int]
    spatial_merge_size: int

    # generation settings
    temperature: float = 0.0
    max_new_tokens: int = 1024
    device = torch.device('cuda:0')

    # kv cache settings
    offload_kv_to_cpu: bool = True
        # True: KV on CPU pinned memory
        # False: KV on GPU (but gpu kv buffer still in use)
    sparse_kv: bool = True
        # True: split video/text KV cache storage, use `SparseKVCacheManager`
        # False: use `KVCacheManager`
    static_sparse_kv: bool = True  # TODO
    dynamic_sparse_kv: bool = True  # TODO



    def __init__(self, model_type: str = 'qwen3vl-8b', **kwargs):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
        self.model_type = model_type
        self.model_path = MODEL_REGISTRY[model_type]
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.dtype = torch.bfloat16
        self.eos_token_id = (151645, 151643)
        self.spatial_merge_size = 2

        self.print_settings()

    
    def print_settings(self):
        print("==== Configurations ====")
        print("Model type:", self.model_type)
        print("Temperature:", self.temperature)
        print("Max new tokens:", self.max_new_tokens)
        print("Offload KV to CPU:", self.offload_kv_to_cpu)
        print("Sparse KV:", self.sparse_kv)
        print("========================")

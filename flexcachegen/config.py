import os
import torch
from transformers import AutoConfig

# TODO: currently only support for qwen3vl-8b
MODEL_REGISTRY: dict[str, str] = {
    'qwen3vl-8b': '/data/lyc/models/Qwen3-VL-8B-Instruct',
}

# TODO: currently only support for qwen3vl-8b
pruning_heads_list = {
    '0.1': {0: [], 1: [], 2: [], 3: [7], 4: [3], 5: [4, 6], 6: [1], 7: [2], 8: [2], 9: [1], 10: [], 11: [], 12: [5], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [2, 3, 7]},
    '0.2': {0: [], 1: [], 2: [1], 3: [5, 6, 7], 4: [1, 3, 6], 5: [2, 4, 5, 6], 6: [1, 5, 7], 7: [2, 3, 4, 7], 8: [2, 5], 9: [1, 6], 10: [2], 11: [], 12: [5], 13: [5], 14: [5], 15: [5], 16: [], 17: [2], 18: [], 19: [4], 20: [6], 21: [0], 22: [6], 23: [], 24: [], 25: [1, 4], 26: [2], 27: [0, 5], 28: [7], 29: [4], 30: [5], 31: [], 32: [5, 6, 7], 33: [0], 34: [2, 6], 35: [0, 1, 2, 3, 7]},
    '0.3': {0: [], 1: [3], 2: [1, 4], 3: [0, 4, 5, 6, 7], 4: [1, 2, 3, 6], 5: [2, 4, 5, 6], 6: [0, 1, 4, 5, 7], 7: [2, 3, 4, 7], 8: [2, 3, 5, 6], 9: [1, 6], 10: [2], 11: [7], 12: [5], 13: [5, 7], 14: [5, 7], 15: [5, 6], 16: [6], 17: [2], 18: [0], 19: [4], 20: [6], 21: [0], 22: [3, 6, 7], 23: [0, 4], 24: [0], 25: [0, 1, 4, 5], 26: [1, 2], 27: [0, 2, 5, 6], 28: [6, 7], 29: [4, 5, 7], 30: [5, 7], 31: [4, 6], 32: [4, 5, 6, 7], 33: [0, 6], 34: [2, 3, 6], 35: [0, 1, 2, 3, 7]},
    '0.4': {0: [0, 4, 5], 1: [3], 2: [0, 1, 4], 3: [0, 1, 4, 5, 6, 7], 4: [1, 2, 3, 4, 5, 6], 5: [2, 4, 5, 6, 7], 6: [0, 1, 4, 5, 7], 7: [1, 2, 3, 4, 6, 7], 8: [2, 3, 4, 5, 6], 9: [1, 6], 10: [2, 3, 4, 5], 11: [7], 12: [5], 13: [5, 7], 14: [0, 5, 7], 15: [5, 6], 16: [6], 17: [2], 18: [0], 19: [1, 2, 4, 5], 20: [6], 21: [0, 3], 22: [3, 6, 7], 23: [0, 4, 6], 24: [0], 25: [0, 1, 3, 4, 5], 26: [1, 2], 27: [0, 2, 5, 6], 28: [6, 7], 29: [4, 5, 7], 30: [0, 3, 5, 7], 31: [4, 6], 32: [3, 4, 5, 6, 7], 33: [0, 6], 34: [2, 3, 6], 35: [0, 1, 2, 3, 4, 7]},
    '0.5': {0: [0, 4, 5], 1: [3], 2: [0, 1, 4, 6, 7], 3: [0, 1, 4, 5, 6, 7], 4: [0, 1, 2, 3, 4, 5, 6], 5: [2, 4, 5, 6, 7], 6: [0, 1, 4, 5, 7], 7: [1, 2, 3, 4, 6, 7], 8: [2, 3, 4, 5, 6], 9: [0, 1, 6], 10: [0, 2, 3, 4, 5], 11: [5, 7], 12: [2, 5, 6], 13: [1, 5, 7], 14: [0, 2, 5, 7], 15: [5, 6], 16: [2, 6], 17: [2], 18: [0], 19: [1, 2, 4, 5], 20: [0, 6], 21: [0, 3], 22: [2, 3, 4, 5, 6, 7], 23: [0, 4, 6], 24: [0], 25: [0, 1, 3, 4, 5], 26: [0, 1, 2, 3], 27: [0, 2, 5, 6], 28: [1, 6, 7], 29: [4, 5, 7], 30: [0, 1, 3, 5, 7], 31: [4, 5, 6], 32: [1, 3, 4, 5, 6, 7], 33: [0, 4, 6], 34: [2, 3, 6], 35: [0, 1, 2, 3, 4, 6, 7]},
    '0.6': {0: [0, 4, 5], 1: [1, 3, 5, 6], 2: [0, 1, 2, 4, 6, 7], 3: [0, 1, 4, 5, 6, 7], 4: [0, 1, 2, 3, 4, 5, 6], 5: [2, 4, 5, 6, 7], 6: [0, 1, 4, 5, 6, 7], 7: [1, 2, 3, 4, 6, 7], 8: [2, 3, 4, 5, 6], 9: [0, 1, 2, 6], 10: [0, 2, 3, 4, 5], 11: [1, 4, 5, 7], 12: [0, 1, 2, 3, 5, 6], 13: [1, 5, 7], 14: [0, 2, 5, 7], 15: [5, 6], 16: [2, 3, 6], 17: [2, 7], 18: [0], 19: [1, 2, 4, 5], 20: [0, 6], 21: [0, 3], 22: [2, 3, 4, 5, 6, 7], 23: [0, 4, 5, 6], 24: [0], 25: [0, 1, 3, 4, 5], 26: [0, 1, 2, 3], 27: [0, 1, 2, 5, 6], 28: [1, 6, 7], 29: [4, 5, 7], 30: [0, 1, 3, 5, 7], 31: [4, 5, 6], 32: [1, 3, 4, 5, 6, 7], 33: [0, 2, 4, 6, 7], 34: [1, 2, 3, 6], 35: [0, 1, 2, 3, 4, 6, 7]},
    '0.7': {0: [0, 4, 5], 1: [0, 1, 3, 4, 5, 6], 2: [0, 1, 2, 4, 6, 7], 3: [0, 1, 2, 4, 5, 6, 7], 4: [0, 1, 2, 3, 4, 5, 6, 7], 5: [0, 1, 2, 4, 5, 6, 7], 6: [0, 1, 3, 4, 5, 6, 7], 7: [1, 2, 3, 4, 6, 7], 8: [2, 3, 4, 5, 6, 7], 9: [0, 1, 2, 6, 7], 10: [0, 2, 3, 4, 5], 11: [1, 4, 5, 7], 12: [0, 1, 2, 3, 5, 6], 13: [1, 3, 5, 6, 7], 14: [0, 2, 5, 6, 7], 15: [1, 4, 5, 6], 16: [2, 3, 6], 17: [0, 2, 5, 7], 18: [0], 19: [1, 2, 4, 5], 20: [0, 1, 6], 21: [0, 1, 3], 22: [0, 2, 3, 4, 5, 6, 7], 23: [0, 4, 5, 6], 24: [0], 25: [0, 1, 3, 4, 5], 26: [0, 1, 2, 3], 27: [0, 1, 2, 5, 6], 28: [1, 6, 7], 29: [4, 5, 7], 30: [0, 1, 3, 5, 7], 31: [1, 4, 5, 6], 32: [1, 3, 4, 5, 6, 7], 33: [0, 2, 3, 4, 6, 7], 34: [1, 2, 3, 6], 35: [0, 1, 2, 3, 4, 6, 7]},
    '0.8': {0: [0, 4, 5], 1: [0, 1, 3, 4, 5, 6], 2: [0, 1, 2, 3, 4, 6, 7], 3: [0, 1, 2, 4, 5, 6, 7], 4: [0, 1, 2, 3, 4, 5, 6, 7], 5: [0, 1, 2, 4, 5, 6, 7], 6: [0, 1, 3, 4, 5, 6, 7], 7: [1, 2, 3, 4, 6, 7], 8: [2, 3, 4, 5, 6, 7], 9: [0, 1, 2, 6, 7], 10: [0, 1, 2, 3, 4, 5], 11: [0, 1, 4, 5, 6, 7], 12: [0, 1, 2, 3, 5, 6], 13: [1, 3, 5, 6, 7], 14: [0, 2, 4, 5, 6, 7], 15: [1, 4, 5, 6, 7], 16: [0, 2, 3, 4, 6, 7], 17: [0, 2, 5, 7], 18: [0, 6], 19: [1, 2, 4, 5], 20: [0, 1, 6], 21: [0, 1, 3], 22: [0, 2, 3, 4, 5, 6, 7], 23: [0, 4, 5, 6], 24: [0], 25: [0, 1, 3, 4, 5, 6], 26: [0, 1, 2, 3], 27: [0, 1, 2, 5, 6], 28: [1, 6, 7], 29: [4, 5, 7], 30: [0, 1, 3, 4, 5, 7], 31: [0, 1, 4, 5, 6], 32: [1, 2, 3, 4, 5, 6, 7], 33: [0, 1, 2, 3, 4, 6, 7], 34: [0, 1, 2, 3, 4, 6], 35: [0, 1, 2, 3, 4, 5, 6, 7]},
    '0.9': {0: [0, 2, 3, 4, 5, 7], 1: [0, 1, 2, 3, 4, 5, 6], 2: [0, 1, 2, 3, 4, 6, 7], 3: [0, 1, 2, 4, 5, 6, 7], 4: [0, 1, 2, 3, 4, 5, 6, 7], 5: [0, 1, 2, 3, 4, 5, 6, 7], 6: [0, 1, 2, 3, 4, 5, 6, 7], 7: [1, 2, 3, 4, 5, 6, 7], 8: [0, 2, 3, 4, 5, 6, 7], 9: [0, 1, 2, 5, 6, 7], 10: [0, 1, 2, 3, 4, 5, 7], 11: [0, 1, 2, 4, 5, 6, 7], 12: [0, 1, 2, 3, 4, 5, 6], 13: [0, 1, 3, 4, 5, 6, 7], 14: [0, 2, 4, 5, 6, 7], 15: [1, 4, 5, 6, 7], 16: [0, 1, 2, 3, 4, 5, 6, 7], 17: [0, 2, 3, 4, 5, 7], 18: [0, 1, 4, 6, 7], 19: [0, 1, 2, 3, 4, 5, 7], 20: [0, 1, 2, 5, 6], 21: [0, 1, 3, 5], 22: [0, 2, 3, 4, 5, 6, 7], 23: [0, 1, 4, 5, 6], 24: [0, 4], 25: [0, 1, 3, 4, 5, 6, 7], 26: [0, 1, 2, 3], 27: [0, 1, 2, 3, 5, 6, 7], 28: [0, 1, 3, 4, 5, 6, 7], 29: [1, 2, 3, 4, 5, 7], 30: [0, 1, 3, 4, 5, 6, 7], 31: [0, 1, 2, 3, 4, 5, 6, 7], 32: [0, 1, 2, 3, 4, 5, 6, 7], 33: [0, 1, 2, 3, 4, 5, 6, 7], 34: [0, 1, 2, 3, 4, 5, 6, 7], 35: [0, 1, 2, 3, 4, 5, 6, 7]},
    '1.0': {
        i: list(range(8)) for i in range(36)
    }
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
    offload_kv_to_cpu: bool = False
        # True: KV on CPU pinned memory
        # False: KV on GPU (but gpu kv buffer still in use)
    sparse_kv: bool = True
        # True: split video/text KV cache storage, use `SparseKVCacheManager`
        # False: use `KVCacheManager`
    static_sparse_threshold: str | None = '0.5'
        # if not None, apply static sparsity with the given threshold
        # (e.g., '0.2' means pruning heads with visual score less than 0.2)
    static_sparse_prune_heads: dict[int, list[int]] | None = None  # {layer_idx: [head_indices]}
        # inferred from `static_sparse_threshold` when initialized
    
    dynamic_sparse_threshold: str | None = None  # TODO



    def __init__(self, model_type: str = 'qwen3vl-8b', **kwargs):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
        self.model_type = model_type
        self.model_path = MODEL_REGISTRY[model_type]
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.dtype = torch.bfloat16
        self.eos_token_id = (151645, 151643)
        self.spatial_merge_size = 2

        if self.static_sparse_threshold is not None:
            if self.static_sparse_threshold not in pruning_heads_list:
                raise ValueError(f"Unknown static_sparse_threshold '{self.static_sparse_threshold}'. Available: {list(pruning_heads_list.keys())}")
            self.static_sparse_prune_heads = pruning_heads_list[self.static_sparse_threshold]

        self.print_settings()

    
    def print_settings(self):
        print(f"{' '+'Configurations' + ' ':=^50}")
        print("Model type:        ", self.model_type)
        print("Temperature:       ", self.temperature)
        print("Max new tokens:    ", self.max_new_tokens)
        print("Offload KV to CPU: ", self.offload_kv_to_cpu)
        print("Sparse KV:         ", self.sparse_kv)
        if self.static_sparse_threshold is not None:
            print(f"  Static sparse threshold: {self.static_sparse_threshold}")
            total_pruned = sum(len(v) for v in self.static_sparse_prune_heads.values())
            total_heads = self.hf_config.text_config.num_key_value_heads * self.hf_config.text_config.num_hidden_layers
            print(f"  Pruning {total_pruned}/{total_heads} heads ({total_pruned/total_heads:.1%} sparsity)")
        print("="*50)

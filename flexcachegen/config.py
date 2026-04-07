import os
import torch
from transformers import AutoConfig

# TODO: currently only support for qwen3vl-8b
MODEL_REGISTRY: dict[str, str] = {
    'qwen3vl-8b': '/data/lyc/models/Qwen3-VL-8B-Instruct',
}

class Config:
    model_type: str

    # model settings (inferred from model_type)
    model_path: str
    hf_config: AutoConfig
    dtype: torch.dtype
    eos_token_id: tuple[int]

    # generation settings
    temperature: float = 0.0
    max_new_tokens: int = 1024
    device = torch.device('cuda:0')

    # kv cache settings
    offload_kv_to_cpu: bool = True  # True: KV on CPU pinned memory, decode via DMA; False: KV on GPU, zero-copy


    def __init__(self, model_type: str = 'qwen3vl-8b', **kwargs):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(MODEL_REGISTRY.keys())}")
        self.model_type = model_type
        self.model_path = MODEL_REGISTRY[model_type]
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.dtype = torch.bfloat16
        self.eos_token_id = (151645, 151643)

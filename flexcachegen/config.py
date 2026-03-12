import os
import torch
from transformers import AutoConfig


class Config:
    # model settings
    device = 'cuda:0'

    # generation settings
    temperature: float = 0.0
    max_new_tokens: int = 1024

    # sparsity settings
    sparsity_strategy: str = 'after-prefill' # ['none', 'before-prefill', 'after-prefill']


    def __init__(self, model_path: str, **kwargs):
        # TODO: currently only support for qwen3vl-8b
        self.model_path = model_path
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        
        self.dtype = torch.bfloat16
        self.spatial_merge_size = 2
        self.eos_token_id = (151645, 151643)


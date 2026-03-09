import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    # model settings
    model_path: str
    hf_config: AutoConfig

    # generation settings
    temperature: float = 0.0
    max_new_tokens: int = 256
    eos_token_id: tuple[int] = (151645, 151643)
    device = torch.device('cuda:1')


    # unused
    # max_num_batched_tokens: int = 16384
    # max_num_seqs: int = 512
    # max_model_len: int = 4096
    # gpu_memory_utilization: float = 0.9
    # tensor_parallel_size: int = 1
    
    # kvcache_block_size: int = 256
    # num_kvcache_blocks: int = -1

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        # for field in self.__dataclass_fields__.values():
        #     if field.name in kwargs:
        #         setattr(self, field.name, kwargs[field.name])
        #     else:
        #         setattr(self, field.name, field.default)
        

    # def __post_init__(self):
    #     assert os.path.isdir(self.model_path)
    #     assert self.kvcache_block_size % 256 == 0
    #     assert 1 <= self.tensor_parallel_size <= 8
    #     self.hf_config = AutoConfig.from_pretrained(self.model_path)
    #     self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
    #     assert self.max_num_batched_tokens >= self.max_model_len

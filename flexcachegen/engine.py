import torch
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm

import sys
from torchinfo import summary

from transformers import AutoProcessor

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.utils import get_tensor_size, print_cuda_memory_usage, print_duration


class VLMEngine:
    '''
    VLMEngine controls the generation process, including computation stage and kv cache IO.
    '''

    def __init__(self, model_path, **kwargs):
        # config
        # config_fields = {field.name for field in fields(Config)}
        # config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # self.config = Config(model_path, **config_kwargs)
        self.config = Config(model_path)
        # kv cache manager
        self.kv_cache_manager = KVCacheManager(self.config)
        # model
        self.model = Qwen3VLModel(self.config, self.kv_cache_manager).to(self.config.device)
        self.processor = AutoProcessor.from_pretrained(self.config.model_path, use_fast=True)
        self.num_hidden_layers = self.config.hf_config.text_config.num_hidden_layers


    def process_input(
        self,
        video_path: str,
        question: str,
    ):
        """
        process input video and question for generation.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

    
    def is_finished(self, output_ids: list[int]) -> bool:
        return output_ids[-1] in self.config.eos_token_id or len(output_ids) >= self.config.max_new_tokens
    

    def print_inputs_size(self, inputs_dict):
        total_bytes = 0
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                v_bytes = v.nelement() * v.element_size()
                total_bytes += v_bytes
                print(f"Input key: {k}, type: {type(v)}, shape: {v.shape}, dtype: {v.dtype}, size: {v_bytes / (1024 ** 2):.2f} MB")

        total_mb = total_bytes / (1024 ** 2)
        print(f"Total inputs size: {total_mb:.2f} MB")

    @torch.inference_mode()
    def generate_single(self, video_path: str, question: str):
        output_ids = []

        # process input
        print('='* 20 + ' Process Input ' + '='*20)
        t = perf_counter()
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.config.device)
        prompt_len = inputs["input_ids"].shape[1]
        print_duration(t, perf_counter())
        self.print_inputs_size(inputs)
        print_cuda_memory_usage(self.config.device)
        
        # encoding step
        print('='* 20 + ' Encoding ' + '='*20)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        t = perf_counter()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            hidden_states = self.model.encoding(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values_videos=inputs["pixel_values_videos"],
                video_grid_thw=inputs["video_grid_thw"]
            )
        print_duration(t, perf_counter())
        print(f'hidden_states size: {get_tensor_size(hidden_states)}')
        print_cuda_memory_usage(self.config.device)

        # prefill stage
        print('='* 20 + ' Prefill ' + '='*20)
        t = perf_counter()
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(True, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
            hidden_states = self.model.merge_visual_features(hidden_states, layer_idx)

        token_id, logits = self.model.output_head(hidden_states)
        output_ids.append(token_id)
        print_duration(t, perf_counter())
        print(f'hidden_states size: {get_tensor_size(hidden_states)}')
        print(f"first token_id={token_id}")
        print_cuda_memory_usage(self.config.device)

        # decoding stage
        print('='* 20 + ' Decoding ' + '='*20)
        t = perf_counter()
        while not self.is_finished(output_ids):
            hidden_states = self.model.text_embed(token_id)
            cur_pos_id = prompt_len + len(output_ids) - 1
            self.model.set_rotary_pos_emb(hidden_states, cur_pos_id)

            for layer_idx in range(self.num_hidden_layers):
                hidden_states = self.model.attention(False, hidden_states, layer_idx)
                hidden_states = self.model.mlp(hidden_states, layer_idx)

            token_id, logits = self.model.output_head(hidden_states)
            output_ids.append(token_id)

            print(f"step {len(output_ids)}: token_id={token_id}")

        print_duration(t, perf_counter())
        print_cuda_memory_usage(self.config.device)

        # clean up kv cache
        self.kv_cache_manager.clear()

        print_cuda_memory_usage(self.config.device)

        # return output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
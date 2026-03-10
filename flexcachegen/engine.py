import torch
from time import perf_counter
from tqdm.auto import tqdm

from transformers import AutoProcessor

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.utils import get_tensor_size, print_cuda_memory_usage, print_duration


class VLMEngine:
    '''
    VLMEngine controls the generation process, scheduling unimplemented.
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

    @print_duration
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
    

    @print_duration
    def prefill(self, hidden_states: torch.Tensor):
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(True, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
            hidden_states = self.model.merge_visual_features(hidden_states, layer_idx)
        token_id, logits = self.model.output_head(hidden_states)
        return token_id, logits
    

    @print_duration
    def decoding(self, token_id: int, cur_pos_id: int):
        hidden_states = self.model.text_embed(token_id)
        self.model.set_rotary_pos_emb(hidden_states, cur_pos_id)
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(False, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
        token_id, logits = self.model.output_head(hidden_states)
        return token_id, logits


    @torch.inference_mode()
    @print_duration
    def generate_single(self, video_path: str, question: str):
        output_ids = []

        # 1. process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.config.device)
        prompt_len = inputs["input_ids"].shape[1]
        print_cuda_memory_usage(self.config.device)
        
        # 2. encoding stage
        hidden_states = self.model.encoding(inputs)
        print_cuda_memory_usage(self.config.device)

        # 3. prefill stage
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)
        print_cuda_memory_usage(self.config.device)

        # 4. decoding stage
        while not self.is_finished(output_ids):
            cur_pos_id = prompt_len + len(output_ids) - 1
            token_id, logits = self.decoding(token_id, cur_pos_id)
            output_ids.append(token_id)
        print_cuda_memory_usage(self.config.device)

        # 5. clean up kv cache
        self.kv_cache_manager.clear()
        print_cuda_memory_usage(self.config.device)

        # 6. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(output_ids)
        print(output_text)

        return output_text
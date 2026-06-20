import torch
from time import perf_counter
from tqdm.auto import tqdm

from transformers import AutoProcessor
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("qwen is not installed. Please install qwen-vl-utils to use this model.")

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.models.qwen25vl import Qwen25VLModel
from flexcachegen.utils import get_tensor_size, print_cuda_memory_usage, print_duration


class VLMEngine:
    '''
    VLMEngine controls the generation process, scheduling logic is unimplemented.
    Only support single GPU single batch inference now.
    '''

    def __init__(self, model_path, **kwargs):
        # model type
        if "qwen2.5" in model_path.lower():
            self.model_type = "qwen2.5"
        elif "qwen3" in model_path.lower():
            self.model_type = "qwen3"
        else:
            self.model_type = "unknown"
        # config
        # config_fields = {field.name for field in fields(Config)}
        # config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # self.config = Config(model_path, **config_kwargs)
        self.config = Config(model_path)
        # kv cache manager
        self.kv_cache_manager = KVCacheManager(self.config)
        # model
        if self.model_type == "qwen2.5":
            self.model = Qwen25VLModel(self.config, self.kv_cache_manager).to(self.config.device)
        elif self.model_type == "qwen3":
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
        if self.model_type == "qwen2.5":
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        elif self.model_type == "qwen3":
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
            if self.model_type == "qwen3":
                hidden_states = self.model.merge_visual_features(hidden_states, layer_idx)
        token_id, logits = self.model.output_head(hidden_states)
        return token_id, logits
    

    def decoding(self, token_id: int, cur_pos_idx: int):
        hidden_states = self.model.text_embed(token_id)
        self.model.set_rotary_pos_emb(hidden_states, cur_pos_idx)
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

        print(f"{inputs.input_ids.shape=}")
        
        # 2. encoding stage
        hidden_states = self.model.encoding(inputs)
        print_cuda_memory_usage(self.config.device)

        # 3. prefill stage
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)
        print_cuda_memory_usage(self.config.device)

        # 4. decoding stage
        t = perf_counter()
        while not self.is_finished(output_ids):
            cur_pos_idx = prompt_len + len(output_ids) - 1
            token_id, logits = self.decoding(token_id, cur_pos_idx)
            output_ids.append(token_id)
        print(f"[decoding] Duration: {perf_counter() - t:.2f} seconds")
        print_cuda_memory_usage(self.config.device)

        # 5. clean up kv cache
        self.kv_cache_manager.clear()
        print_cuda_memory_usage(self.config.device)

        # 6. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
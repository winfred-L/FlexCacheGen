import torch
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm

from transformers import AutoProcessor

from flexcachegen.kvcache import KVCacheManager
from flexcachegen.config import Config
from flexcachegen.models.qwen3vl import Qwen3VLModel


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
        

    def generate_single(self, video_path: str, question: str):
        output_ids = []

        # process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.config.device)
        
        # encoding step
        hidden_states = self.model.encoding(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"]
        )

        # prefill stage
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(True, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
            hidden_states = self.model.merge_visual_features(hidden_states, layer_idx)

        token_id, logits = self.model.output_head(hidden_states)
        output_ids.append(token_id)

        # decoding stage
        while not self.is_finished(output_ids):
            hidden_states = self.model.text_embed(token_id)
            self.model.set_rotary_pos_emb(hidden_states, len(output_ids))

            for layer_idx in range(self.num_hidden_layers):
                hidden_states = self.model.attention(False, hidden_states, layer_idx)
                hidden_states = self.model.mlp(hidden_states, layer_idx)

            token_id, logits = self.model.output_head(hidden_states)
            output_ids.append(token_id)

            print(f"step {len(output_ids)}: token_id={token_id}")

        # clean up kv cache
        self.kv_cache_manager.clear()

        # return output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
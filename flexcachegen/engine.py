import torch
from time import perf_counter
from tqdm.auto import tqdm

from transformers import AutoProcessor

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager, SparseKVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.utils import get_tensor_size, print_cuda_memory_usage, print_duration


class VLMEngine:
    '''
    VLMEngine controls the generation process, scheduling logic is unimplemented.
    Only support single GPU single batch inference now.
    '''

    def __init__(self, model_type='qwen3vl-8b', **kwargs):
        # config
        self.config = Config(model_type)
        # kv cache manager
        if self.config.sparse_kv:
            self.kv_cache_manager = SparseKVCacheManager(self.config)
        else:
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
    

    def decoding(self, token_id: int, cur_pos_idx: int):
        hidden_states = self.model.text_embed(token_id)
        self.model.set_rotary_pos_emb(hidden_states, cur_pos_idx)
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self.model.attention(False, hidden_states, layer_idx)
            hidden_states = self.model.mlp(hidden_states, layer_idx)
        token_id, logits = self.model.output_head(hidden_states)
        return token_id, logits


    @torch.inference_mode()
    def generate_single(self, video_path: str, question: str):
        output_ids = []
        t_start = perf_counter()

        # 1. process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.config.device)
        prompt_len = inputs["input_ids"].shape[1]
        print_cuda_memory_usage(self.config.device)

        print(f"{inputs.input_ids.shape=}")

        # 1.5 extract video info for sparse KV cache
        if isinstance(self.kv_cache_manager, SparseKVCacheManager):
            video_info = self.model.get_video_info(inputs)
            self.kv_cache_manager.set_video_info(video_info, prompt_len)

        # 2. encoding stage
        hidden_states = self.model.encoding(inputs)
        print_cuda_memory_usage(self.config.device)

        # 3. prefill stage
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)
        t_first_token = perf_counter()
        print_cuda_memory_usage(self.config.device)

        # 4. decoding stage
        t_decode_start = perf_counter()
        while not self.is_finished(output_ids):
            cur_pos_idx = prompt_len + len(output_ids) - 1
            token_id, logits = self.decoding(token_id, cur_pos_idx)
            output_ids.append(token_id)
        t_end = perf_counter()
        print(f"[decoding] Duration: {t_end - t_decode_start:.2f} seconds")
        print_cuda_memory_usage(self.config.device)

        # 5. performance metrics
        ttft = t_first_token - t_start
        total_time = t_end - t_start
        num_generated_tokens = len(output_ids)
        num_decode_tokens = num_generated_tokens - 1
        decode_duration = t_end - t_decode_start
        tpot = (decode_duration / num_decode_tokens * 1000) if num_decode_tokens > 0 else 0.0
        throughput = num_generated_tokens / total_time if total_time > 0 else 0.0

        print(f"\n{'=' * 45}")
        print(f" Performance Metrics")
        print(f"{'=' * 45}")
        print(f" TTFT:                {ttft:.2f} s")
        print(f" TPOT:                {tpot:.2f} ms")
        print(f" Generated tokens:    {num_generated_tokens}")
        print(f" Total time:          {total_time:.2f} s")
        print(f" Throughput:          {throughput:.2f} tokens/s")
        print(f"{'=' * 45}\n")

        # 6. clean up kv cache
        self.kv_cache_manager.clear()

        # 7. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
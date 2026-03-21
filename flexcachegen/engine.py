import torch
from time import perf_counter
from tqdm.auto import tqdm

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager, PagedKVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.utils import get_tensor_size, print_cuda_memory_usage, print_duration


class VLMEngine:
    '''
    VLMEngine controls the generation process, scheduling logic is unimplemented.
    Only support single GPU single batch inference now.
    '''

    def __init__(self, model_path, **kwargs):
        # config
        # config_fields = {field.name for field in fields(Config)}
        # config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # self.config = Config(model_path, **config_kwargs)
        self.config = Config(model_path)
        # kv cache manager
        if self.config.paged_kv:
            self.kv_cache_manager = PagedKVCacheManager(self.config)
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
                        # restrict the resolution of individual frames in the video
                        # "min_pixels": 4 * 32 * 32,
                        # "max_pixels": 640 * 32 * 32,
                        # limit the total number of tokens in the video
                        "total_pixels": 100 * 1024 * 32 * 32, # 64k tokens
                        # accept either `fps` or `nframes`
                        # "fps": 2.0,
                        # "nframes": 32, #2048,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        # each video returns as (video_tensor, video_metadata)
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False, # avoid duplicate resizing
            **video_kwargs
        )
        return inputs
    
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": video_path,
        #                 "max_pixels": 360 * 420,
        #                 "fps": 1.0,
        #             },
        #             {"type": "text", "text": question},
        #         ],
        #     }
        # ]
        # inputs = self.processor.apply_chat_template(
        #     messages,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_dict=True,
        #     return_tensors="pt"
        # )
        # return inputs

    
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

    def decoding_pipelined(self, token_id: int, cur_pos_idx: int):
        """Pipeline decode: overlap DMA of next layer with attention+MLP of current layer."""
        hidden_states = self.model.text_embed(token_id)
        self.model.set_rotary_pos_emb(hidden_states, cur_pos_idx)
        kvcm = self.kv_cache_manager

        for layer_idx in range(self.num_hidden_layers):
            # Phase 1: QKV projection
            q, k, v, residual = self.model.attention_qkv(hidden_states, layer_idx)

            # Phase 2: prepare KV cache (finalize pre-loaded DMA or sync load)
            gpu_k, gpu_v, block_table, cache_seqlens = kvcm.prepare_layer_for_decode(layer_idx, q)

            # Phase 3: start async pre-load for next layer (overlaps with Phase 4+5)
            if layer_idx + 1 < self.num_hidden_layers:
                kvcm.predict_and_start_dma(layer_idx + 1, q)

            # Phase 4: Attention (parallel with DMA stream)
            hidden_states = self.model.attention_compute(
                q, k, v, residual, layer_idx, gpu_k, gpu_v, block_table, cache_seqlens)

            # Phase 5: MLP (still parallel with DMA)
            hidden_states = self.model.mlp(hidden_states, layer_idx)

            # Phase 6: write back new token to CPU
            kvcm.update_after_decode(layer_idx)

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
        # print_cuda_memory_usage(self.config.device)
        print(f"{inputs.input_ids.shape=}")

        # 1.5. set video info to kv cache manager, used for kv sparsity eviction strategy
        self.kv_cache_manager.set_video_info(self.model.get_video_info(inputs))
        
        # 2. encoding stage
        hidden_states = self.model.encoding(inputs)
        # print_cuda_memory_usage(self.config.device)

        # 3. prefill stage
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)
        # print_cuda_memory_usage(self.config.device)

        # 4. decoding stage
        t = perf_counter()
        if self.config.pipeline_decode and self.config.paged_kv and self.config.block_topk_ratio is not None:
            self.kv_cache_manager.enable_pipeline()
            decode_fn = self.decoding_pipelined
        else:
            decode_fn = self.decoding
        while not self.is_finished(output_ids):
            cur_pos_idx = prompt_len + len(output_ids) - 1
            token_id, logits = decode_fn(token_id, cur_pos_idx)
            output_ids.append(token_id)
        print(f"[decoding] Duration: {perf_counter() - t:.2f} seconds")
        # print_cuda_memory_usage(self.config.device)

        # 5. clean up kv cache
        self.kv_cache_manager.clear()
        # print_cuda_memory_usage(self.config.device)

        # 6. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
import torch
from time import perf_counter

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager, SparseKVCacheManager
from flexcachegen.models.qwen3vl import Qwen3VLModel
from flexcachegen.utils import format_bytes


class VLMEngine:
    '''
    VLMEngine controls the generation process, scheduling logic is unimplemented.
    Only support single GPU single batch inference now.
    '''

    def __init__(self, model_type='qwen3vl-8b', **kwargs):
        t_init = perf_counter()
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
        torch.cuda.synchronize()
        self.model_memory = torch.cuda.memory_allocated(self.config.device)
        print(f"Model loaded in {perf_counter() - t_init:.2f} s\n")

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
                        "resized_height": 16 * 32,  # VideoInfo.H_len=16
                        "resized_width": 16 * 32,   # VideoInfo.W_len=16
                        # restrict the resolution of individual frames in the video
                        # "min_pixels": 275 * 32 * 32,
                        # "max_pixels": 275 * 32 * 32,
                        # limit the total number of tokens in the video
                        # "total_pixels": 100 * 1024 * 32 * 32, # 64k tokens
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
        print(video_kwargs)

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

        ### 注：processor.apply_chat_template走的是hf的实现，不支持resized_height/resized_width, 只支持smart resize
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": video_path,
        #                 "resized_height": 16 * 32,
        #                 "resized_width": 16 * 32,
        #                 # "max_pixels": 360 * 420,
        #                 # "fps": 1.0,
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
        """Generate without any profiling overhead."""
        output_ids = []

        # 1. process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(self.config.device)
        prompt_len = inputs["input_ids"].shape[1]

        if isinstance(self.kv_cache_manager, SparseKVCacheManager):
            video_info = self.model.get_video_info(inputs)
            self.kv_cache_manager.set_video_info(video_info, prompt_len)

        # 2. encoding stage
        hidden_states = self.model.encoding(inputs)

        # 3. prefill stage
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)

        # 4. decoding stage
        while not self.is_finished(output_ids):
            cur_pos_idx = prompt_len + len(output_ids) - 1
            token_id, logits = self.decoding(token_id, cur_pos_idx)
            output_ids.append(token_id)

        # 5. clean up kv cache
        self.kv_cache_manager.clear()

        # 6. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    @torch.inference_mode()
    def generate_single_info(self, video_path: str, question: str):
        """Generate with detailed performance and memory profiling."""
        output_ids = []
        device = self.config.device
        t_start = perf_counter()

        # global peak tracking (never reset)
        torch.cuda.reset_peak_memory_stats(device)

        # 1. process input
        inputs = self.process_input(video_path, question)
        inputs = inputs.to(device)
        prompt_len = inputs["input_ids"].shape[1]

        if isinstance(self.kv_cache_manager, SparseKVCacheManager):
            video_info = self.model.get_video_info(inputs)
            self.kv_cache_manager.set_video_info(video_info, prompt_len)
        t_process = perf_counter()

        # 2. encoding stage
        torch.cuda.synchronize()
        mem_before_encoding = torch.cuda.memory_allocated(device)
        peak_global = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        hidden_states = self.model.encoding(inputs)
        torch.cuda.synchronize()
        peak_encoding = torch.cuda.max_memory_allocated(device) - mem_before_encoding
        peak_global = max(peak_global, torch.cuda.max_memory_allocated(device))
        t_encoding = perf_counter()

        # 3. prefill stage
        torch.cuda.synchronize()
        mem_before_prefill = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        token_id, logits = self.prefill(hidden_states)
        output_ids.append(token_id)
        torch.cuda.synchronize()
        peak_prefill = torch.cuda.max_memory_allocated(device) - mem_before_prefill
        peak_global = max(peak_global, torch.cuda.max_memory_allocated(device))
        t_prefill = perf_counter()

        # snapshot KV cache stats after prefill (before decode modifies seq_len)
        kv_stats = self.kv_cache_manager.get_memory_stats()

        # 4. decoding stage
        torch.cuda.synchronize()
        mem_before_decoding = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        while not self.is_finished(output_ids):
            cur_pos_idx = prompt_len + len(output_ids) - 1
            token_id, logits = self.decoding(token_id, cur_pos_idx)
            output_ids.append(token_id)
        torch.cuda.synchronize()
        peak_decoding = torch.cuda.max_memory_allocated(device) - mem_before_decoding
        peak_global = max(peak_global, torch.cuda.max_memory_allocated(device))
        t_end = perf_counter()

        # 5. print results
        print(f"{' '+'Input' + ' ':=^50}")
        print(f"Video: {video_path}")
        print(f"Question: {question}")

        # 5.1 performance metrics
        ttft = t_prefill - t_start
        total_time = t_end - t_start
        num_generated_tokens = len(output_ids)
        num_decode_tokens = num_generated_tokens - 1
        decode_duration = t_end - t_prefill
        tpot = (decode_duration / num_decode_tokens * 1000) if num_decode_tokens > 0 else 0.0
        throughput = num_generated_tokens / total_time if total_time > 0 else 0.0

        print(f"{' '+'Performance Metrics' + ' ':=^50}")
        print(f" Total time:          {total_time:.2f} s")
        print(f"   Process input:     {t_process - t_start:.2f} s")
        print(f"   Encoding:          {t_encoding - t_process:.2f} s")
        print(f"   Prefill:           {t_prefill - t_encoding:.2f} s")
        print(f"   Decoding:          {decode_duration:.2f} s")
        print(f" TTFT:                {ttft:.2f} s")
        print(f" TPOT:                {tpot:.2f} ms")
        print(f" Prompt tokens:       {prompt_len}")
        print(f" Generated tokens:    {num_generated_tokens}")
        print(f" Throughput:          {throughput:.2f} tokens/s")

        # 5.2 memory report
        kv_total = kv_stats["kv_cache_gpu_bytes"] + kv_stats["kv_cache_cpu_bytes"]
        kv_device = "CPU" if self.config.offload_kv_to_cpu else "GPU"
        n_layers = self.num_hidden_layers

        print(f"{' '+'Memory Usage' + ' ':=^50}")
        print(f" Model weights:       {format_bytes(self.model_memory)}")
        print(f" KV cache (all):      {format_bytes(kv_total)}  ({n_layers} layers, {kv_device})")
        if isinstance(self.kv_cache_manager, SparseKVCacheManager):
            print(f"   Video KV:          {format_bytes(kv_stats['video_kv_bytes'])}")
            print(f"   Text KV:           {format_bytes(kv_stats['text_kv_bytes'])}")
        print(f" GPU KV buffer:       {format_bytes(kv_stats['gpu_buffer_bytes'])}")
        print(f" Peak activation:")
        print(f"   Encoding:          {format_bytes(peak_encoding)}")
        print(f"   Prefill:           {format_bytes(peak_prefill)}")
        print(f"   Decoding:          {format_bytes(peak_decoding)}")
        print(f" Peak global:         {format_bytes(peak_global)}")

        # 6. clean up kv cache
        self.kv_cache_manager.clear()

        # 7. decode output text
        output_text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"{' '+'Output' + ' ':=^50}")
        print(output_text)
        print()

        return output_text
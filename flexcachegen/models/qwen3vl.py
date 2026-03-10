import glob
import os
import torch
from torch import nn
from safetensors.torch import load_file
from typing import Optional
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLTextRotaryEmbedding, Qwen3VLTextRMSNorm

from flash_attn import flash_attn_func, flash_attn_with_kvcache

from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager
from flexcachegen.models.qwen3vl_util import get_video_features, get_placeholder_mask, get_rope_index
from flexcachegen.utils import print_duration



class Qwen3VLTextInputEmbed(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        vocab_size = config.hf_config.text_config.vocab_size
        hidden_size = config.hf_config.text_config.hidden_size
        padding_idx = config.hf_config.text_config.pad_token_id
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)

    def forward(self, token_id: int) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(token_id)
        return inputs_embeds
    

class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_attention_heads = config.hf_config.text_config.num_attention_heads
        self.num_key_value_heads = config.hf_config.text_config.num_key_value_heads
        self.hidden_size = config.hf_config.text_config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_hidden_size = self.head_dim * self.num_key_value_heads
        self.rms_norm_eps = config.hf_config.text_config.rms_norm_eps

        self.input_layernorm = Qwen3VLTextRMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_hidden_size, bias=False)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def forward(
        self,
        is_prefill: bool,
        hidden_states: torch.Tensor,
        kv_cache_manager: KVCacheManager,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        assert batch_size == 1, "Only batch size 1 is supported."

        residual = hidden_states

        # shape: (batch, seq, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)

        # shape: (batch, seq, hidden_size)
        q = self.q_proj(hidden_states)
        # shape: (batch, seq, kv_hidden_size)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # shape: (batch, seq, num_attention_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        # shape: (batch, seq, num_key_value_heads, head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # qk norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # do rotary position embedding
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(
            q, k, cos, sin
        )
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        if is_prefill:
            # store kv to kv cache pool
            kv_cache_manager.gpu_k_buffer[self.layer_idx][:, :seq_len].copy_(k)
            kv_cache_manager.gpu_v_buffer[self.layer_idx][:, :seq_len].copy_(v)
            kv_cache_manager.cache_seqlens[self.layer_idx].fill_(seq_len)
            kv_cache_manager.offload_layer_to_cpu(self.layer_idx)

            # attention computation
            # shape: (batch, seq, num_attention_heads, head_dim)
            attn_output = flash_attn_func(q, k, v, causal=True)
        
        else: # decoding stage
            # retrieve kv to GPU from kv cache pool
            kv_cache_manager.load_layer_to_gpu(self.layer_idx)

            # attention computation
            # flash_attn_with_kvcache() will update kv cache inside
            attn_output = flash_attn_with_kvcache(
                q=q,
                k_cache=kv_cache_manager.gpu_k_buffer[self.layer_idx],
                v_cache=kv_cache_manager.gpu_v_buffer[self.layer_idx],
                k=k,
                v=v,
                cache_seqlens=kv_cache_manager.cache_seqlens[self.layer_idx],
                causal=True
            )

            # store kv to kv cache pool
            kv_cache_manager.offload_layer_to_cpu(self.layer_idx)

        # shape: (batch, seq, hidden_size)
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        hidden_states = residual + attn_output
        return hidden_states


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        hidden_size = config.hf_config.text_config.hidden_size
        intermediate_size = config.hf_config.text_config.intermediate_size
        rms_norm_eps = config.hf_config.text_config.rms_norm_eps
        
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        down = self.down_proj(self.act_fn(gate) * up)
        hidden_states = residual + down
        return hidden_states
    

class Qwen3VLOutputHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        hidden_size = config.hf_config.text_config.hidden_size
        vocab_size = config.hf_config.text_config.vocab_size
        rms_norm_eps = config.hf_config.text_config.rms_norm_eps

        self.norm = Qwen3VLTextRMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[int, torch.Tensor]:
        hidden_states = self.norm(hidden_states)

        last_token_hidden = hidden_states[:, -1:, :]
        last_token_logits = self.lm_head(last_token_hidden)
        last_token_logits = last_token_logits.squeeze(1)

        if self.config.temperature > 0:
            probs = torch.softmax(last_token_logits / self.config.temperature, dim=-1)
            sampled_token_id = torch.multinomial(probs, num_samples=1)
            token_id = sampled_token_id.item()
        else:
            token_id = torch.argmax(last_token_logits, dim=-1).item()

        return token_id, last_token_logits




class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)

    def forward(self):
        raise NotImplementedError("Use specific layers for generating.")


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        # components
        self.input_embed = Qwen3VLTextInputEmbed(config)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config.hf_config.text_config)
        self.layers = nn.ModuleList(
            [Qwen3VLDecoderLayer(config, layer_idx) for layer_idx in range(self.num_hidden_layers)]
        )
        self.output_head = Qwen3VLOutputHead(config)

    def forward(self):
        raise NotImplementedError("Use specific layers for generating.")


class Qwen3VLModel(nn.Module):
    '''
    Computation implementation of Qwen3VLModel, providing computing stage APIs for VLMEngine.
    KV cache IO is managed by KVCacheManager in VLMEngine, not here.
    '''

    rope_deltas = None
    position_embeddings = None

    visual_pos_masks = None
    deepstack_video_embeds = None
    
    @print_duration
    def __init__(self, config: Config, kv_cache_manager: KVCacheManager):
        super().__init__()
        self.config = config
        self.kv_cache_manager = kv_cache_manager

        # components
        self.visual_model = Qwen3VLVisionModel._from_config(config.hf_config.vision_config) # use origin implementation
        self.language_model = Qwen3VLTextModel(config)
        
        # load weights
        weight_files = glob.glob(os.path.join(config.model_path, "*.safetensors"))
        if not weight_files:
            raise RuntimeError(f"No weight files found in {config.model_path}")
        
        state_dict = {}
        for f in weight_files:
            shard = load_file(f, device="cpu")
            state_dict.update(shard)

        visual_state_dict = {
            k.replace("model.visual.", ""): v 
            for k, v in state_dict.items() if k.startswith("model.visual.")
        }
        visual_info = self.visual_model.load_state_dict(visual_state_dict, strict=False)
        assert len(visual_info.missing_keys) == 0, "Some visual model weights are missing!"
        # print(f"Vision Missing Keys: {visual_info.missing_keys}")
        # print(f"Vision Unexpected Keys: {visual_info.unexpected_keys}")

        text_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "")
                if new_k == "embed_tokens.weight":
                    new_k = "input_embed.embed_tokens.weight"
                elif ".input_layernorm." in new_k:
                    new_k = new_k.replace("input_layernorm", "self_attn.input_layernorm")
                elif ".post_attention_layernorm." in new_k:
                    new_k = new_k.replace("post_attention_layernorm", "mlp.post_attention_layernorm")
                elif new_k == "norm.weight":
                    new_k = "output_head.norm.weight"
                text_state_dict[new_k] = v
            elif k == "lm_head.weight":
                text_state_dict["output_head.lm_head.weight"] = v
        text_info = self.language_model.load_state_dict(text_state_dict, strict=False)
        assert len(text_info.missing_keys) == 0, "Some text model weights are missing!"
        # print(f"Text Missing Keys: {text_info.missing_keys}")
        # print(f"Text Unexpected Keys: {text_info.unexpected_keys}")

        # move to gpu
        self.visual_model.to(device=self.config.device).eval() # actually torch.float32 !!!
        self.language_model.to(device=self.config.device, dtype=self.config.dtype).eval()
        
        del state_dict, visual_state_dict, text_state_dict
        torch.cuda.empty_cache()

        print("Qwen3VLModel init done.")

    @print_duration
    def encoding(self, inputs):

        # get inputs
        input_ids = inputs["input_ids"].to(self.config.device)
        attention_mask = inputs["attention_mask"].to(self.config.device)
        pixel_values_videos = inputs["pixel_values_videos"].to(self.config.device)
        video_grid_thw = inputs["video_grid_thw"].to(self.config.device)

        # encode texts into embeddings
        inputs_embeds = self.language_model.input_embed(input_ids)
        inputs_embeds = inputs_embeds.to(device=self.config.device, dtype=self.config.dtype)
        
        # encode videos into embeddings
        video_embeds, deepstack_video_embeds = get_video_features(self.visual_model, pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(device=self.config.device, dtype=self.config.dtype)
        
        # merge video embeddings into text embeddings
        _, video_mask = get_placeholder_mask(
            self.language_model.input_embed, self.config.hf_config,
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # cache deepstack merger args
        self.visual_pos_masks = video_mask[..., 0]
        self.deepstack_video_embeds = deepstack_video_embeds

        # get attention_mask_tensor
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            # Only apply conversion for floating point tensors (inverted masks)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        # set position embeddings
        position_ids, rope_deltas = get_rope_index(
            config=self.config.hf_config, 
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask_tensor,
        )
        self.rope_deltas = rope_deltas
        self.position_embeddings = self.language_model.rotary_emb(inputs_embeds, position_ids)

        return inputs_embeds

    
    def text_embed(self, token_id: int) -> torch.Tensor:
        hidden_states = torch.tensor([[token_id]], device=self.config.device, dtype=torch.long)
        return self.language_model.input_embed.forward(hidden_states)
    

    def set_rotary_pos_emb(self, inputs_embeds: torch.Tensor, cur_pos_id: int):
        inputs_embeds = inputs_embeds.to(device=self.config.device)

        batch_size, seq_length, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        delta = (cur_pos_id + self.rope_deltas).to(inputs_embeds.device)
        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        position_ids = position_ids.add(delta)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        self.position_embeddings = self.language_model.rotary_emb(inputs_embeds, position_ids)

    
    def attention(
        self,
        is_prefill: bool,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.layers[layer_idx].self_attn.forward(
            is_prefill,
            hidden_states,
            self.kv_cache_manager,
            self.position_embeddings,
        ) 


    def mlp(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.layers[layer_idx].mlp.forward(
            hidden_states,
        )


    def merge_visual_features(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        # add visual features to the hidden states of first several decoder layers (only prefill)
        if layer_idx < len(self.deepstack_video_embeds):
            visual_pos_masks = self.visual_pos_masks.to(device=hidden_states.device)
            deepstack_video_embeds = self.deepstack_video_embeds[layer_idx].to(device=hidden_states.device, dtype=hidden_states.dtype)
            local_this = hidden_states[visual_pos_masks, :].clone() + deepstack_video_embeds
            hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


    def output_head(self, hidden_states: torch.Tensor) -> tuple[int, torch.Tensor]:
        return self.language_model.output_head(hidden_states)

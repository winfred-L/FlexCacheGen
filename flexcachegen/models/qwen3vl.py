import glob
import os
import torch
from torch import nn
from safetensors.torch import load_file
from flexcachegen.config import Config
from flexcachegen.kvcache import KVCacheManager

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLTextRotaryEmbedding
from flexcachegen.models.qwen3vl_layer import (
    Qwen3VLTextInputEmbed,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLOutputHead,
)
from flexcachegen.models.qwen3vl_util import get_video_features, get_placeholder_mask, get_rope_index


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)

    def forward():
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
    
    def __init__(self, config: Config, kv_cache_manager: KVCacheManager):
        super().__init__()
        self.config = config
        self.kv_cache_manager = kv_cache_manager

        # components
        self.visual_model = Qwen3VLVisionModel(config.hf_config.vision_config) # use origin implementation
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
        
        del state_dict, visual_state_dict, text_state_dict


    def encoding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
    ):
        # encode texts into embeddings
        inputs_embeds = self.language_model.input_embed(input_ids)
        
        # encode videos into embeddings
        video_embeds, deepstack_video_embeds = get_video_features(self.visual_model, pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.dtype)
        
        # merge video embeddings into text embeddings
        _, video_mask = get_placeholder_mask(
            self.language_model.input_embed, self.config.hf_config,
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # set deepstack merger args
        self.visual_pos_masks = video_mask[..., 0]
        self.deepstack_video_embeds = deepstack_video_embeds

        # set position embeddings
        position_ids, rope_deltas = get_rope_index(
            config=self.config.hf_config, 
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        self.rope_deltas = rope_deltas
        self.position_embeddings = self.language_model.rotary_emb(inputs_embeds, position_ids)

        return inputs_embeds

    
    def text_embed(self, token_id: int) -> torch.Tensor:
        return self.language_model.input_embed.forward(token_id)
    

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
            visual_pos_masks = self.visual_pos_masks#.to(device=self.config.device)
            deepstack_video_embeds = self.deepstack_video_embeds#.to(device=self.config.device, dtype=hidden_states.dtype)
            local_this = hidden_states[visual_pos_masks, :].clone() + deepstack_video_embeds
            hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


    def output_head(self, hidden_states: torch.Tensor) -> tuple[int, torch.Tensor]:
        return self.language_model.output_head(hidden_states)

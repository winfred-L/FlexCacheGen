from typing import Optional
import torch
from torch import nn
from flexcachegen.kvcache import KVCacheManager
from flexcachegen.config import Config
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
from flash_attn import flash_attn_func, flash_attn_with_kvcache


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
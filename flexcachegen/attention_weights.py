"""Attention weight collection during decode for visualization.

Provides:
  - compute_attention_weights(): manually compute softmax(QK^T/sqrt(d)) with GQA
  - AttentionWeightCollector: organizes per-step, per-layer collection and saving
"""

import json
import math
from pathlib import Path

import torch


@torch.no_grad()
def compute_attention_weights(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute attention weights = softmax(Q @ K^T / sqrt(d)) for one decode token.

    Handles GQA by repeating each KV head to match its Q-head group.

    flash_attn_with_kvcache writes the new token's K at position cache_seqlens
    and then computes attention over all cache_seqlens+1 keys. This function
    must be called AFTER flash_attn so k_cache has the new K at that position.

    Args:
        q:            [1, 1, num_q_heads, head_dim] -- post-RoPE query.
        k_cache:      [1, max_seq_len, num_kv_heads, head_dim] -- GPU buffer
                       with the new token's K already written at cache_seqlens.
        cache_seqlens: valid entries BEFORE the new token was appended.
        num_q_heads:  e.g. 32.
        num_kv_heads: e.g. 8.

    Returns:
        weights: [num_q_heads, cache_seqlens + 1] float32 tensor.
    """
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5
    group_size = num_q_heads // num_kv_heads

    q_heads = q[0, 0]  # [num_q_heads, head_dim]

    total_len = cache_seqlens + 1

    # k_cache[:, :total_len] includes original cache plus the new token's K
    k_all = k_cache[0, :total_len]  # [total_len, num_kv_heads, head_dim]

    # GQA: repeat each KV head group_size times to match Q head count
    k_expanded = k_all.repeat_interleave(group_size, dim=1)  # [total_len, num_q_heads, head_dim]

    # Per-head dot product: (q_heads[h] . k_expanded[s, h]) / sqrt(d)
    scores = torch.einsum("hd,shd->hs", q_heads, k_expanded) * scale  # [num_q_heads, total_len]

    weights = torch.softmax(scores.float(), dim=-1)  # float32 softmax for numerical stability
    return weights  # [num_q_heads, total_len]


class AttentionWeightCollector:
    """Collects attention weights per decode step, per layer, per head.

    Usage:
        collector = AttentionWeightCollector(config, output_dir)
        # attach to each attention layer:
        for layer in model.language_model.layers:
            layer.self_attn.attn_weight_collector = collector

        for each decode step:
            collector.start_step()
            # ... model runs, each attention layer calls collector.add_layer(...)
            collector.finish_step()

        collector.save_metadata()
    """

    def __init__(self, config, output_dir: str):
        self.enabled = config.save_attention_weights
        if not self.enabled:
            return

        self.output_dir = Path(output_dir)
        self.weights_dir = self.output_dir / "attn_weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.current_step_data: dict[int, torch.Tensor] = {}
        self.steps_completed: int = 0
        self.video_len: int = 0
        self.text_prefix_len: int = 0
        self.metadata: dict = {
            "model_type": getattr(config, "model_type", "unknown"),
            "num_hidden_layers": config.hf_config.text_config.num_hidden_layers,
            "num_attention_heads": config.hf_config.text_config.num_attention_heads,
            "num_key_value_heads": config.hf_config.text_config.num_key_value_heads,
            "video_len": 0,
            "text_prefix_len": 0,
            "tokens_per_frame": 0,
            "steps_info": {},
        }

    def set_sequence_info(self, video_len: int, text_prefix_len: int, tokens_per_frame: int = 0):
        """Record video/text token boundary for plot background spans.

        In the GPU buffer, video tokens occupy [0, video_len) and text tokens
        (prefix + generated) occupy [video_len, total_len). This division is
        known after prefill when SparseKVCacheManager separates video/text KV.

        Args:
            video_len: Number of video tokens in the KV cache.
            text_prefix_len: Number of text tokens before decode starts.
            tokens_per_frame: H*W spatial tokens per temporal frame (0 if
                              unknown). Used to draw per-frame alternating
                              background bands in the video region.
        """
        if not self.enabled:
            return
        self.video_len = video_len
        self.text_prefix_len = text_prefix_len
        self.metadata["video_len"] = video_len
        self.metadata["text_prefix_len"] = text_prefix_len
        self.metadata["tokens_per_frame"] = tokens_per_frame
        print(f"[collector] set_sequence_info: video_len={video_len}, "
              f"text_prefix_len={text_prefix_len}, tokens_per_frame={tokens_per_frame}")

    def start_step(self):
        if not self.enabled:
            return
        self.current_step_data = {}

    def add_layer(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        cache_seqlens: int,
        num_q_heads: int,
        num_kv_heads: int,
    ):
        """Compute and store attention weights for one layer.

        Must be called after flash_attn_with_kvcache so k_cache has the new K.
        Result is moved to CPU immediately to avoid accumulating GPU memory.
        """
        if not self.enabled:
            return
        weights = compute_attention_weights(
            q, k_cache, cache_seqlens, num_q_heads, num_kv_heads
        )
        self.current_step_data[layer_idx] = weights.cpu()

    def finish_step(self):
        """Save per-step data to disk as a stacked tensor."""
        if not self.enabled:
            return
        num_layers = len(self.current_step_data)
        if num_layers == 0:
            return

        layer_keys = sorted(self.current_step_data.keys())
        tensors = [self.current_step_data[k] for k in layer_keys]
        stacked = torch.stack(tensors, dim=0)  # [num_layers, num_q_heads, seq_len]

        seq_len = stacked.shape[-1]
        self.metadata["steps_info"][str(self.steps_completed)] = {"seq_len": seq_len}

        save_path = self.weights_dir / f"step_{self.steps_completed:04d}.pt"
        torch.save(stacked, save_path)

        self.steps_completed += 1
        self.current_step_data = {}

    def save_metadata(self):
        if not self.enabled:
            return
        self.metadata["num_steps"] = self.steps_completed
        meta_path = self.weights_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Attention weights saved to {self.weights_dir} ({self.steps_completed} steps)")

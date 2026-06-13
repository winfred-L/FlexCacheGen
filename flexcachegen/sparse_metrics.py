"""Sparse-pruning metrics collection during decode for visualization.

Collects per-step, per-layer data from _load_layer_quest:
  - text_scores_sorted: sorted text Q·K scores [T]
  - page_scores: per-page Quest upper-bound scores [P]
  - page_cutoff / gate_cutoff: percentile thresholds derived from text scores
  - global_score: global video upper bound (for gating comparison)
  - selected_mask: boolean mask of kept pages [P]
  - video_tokens_used: actual video tokens loaded
"""

import json
from pathlib import Path

import torch


class SparseMetricsCollector:
    """Collects sparse-pruning metrics per decode step, per layer.

    Attached to SparseKVCacheManager. _load_layer_quest calls add_layer()
    when self.collector is not None.
    """

    def __init__(self, config, output_dir: str):
        self.enabled = config.save_sparse_metrics
        if not self.enabled:
            return

        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "sparse_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.dynamic_sparse_threshold = config.dynamic_sparse_threshold
        self.video_gating_threshold = config.video_gating_threshold
        self.page_size = config.page_size

        self.current_step_data: dict[int, dict] = {}
        self.steps_completed: int = 0
        self.metadata: dict = {
            "model_type": getattr(config, "model_type", "unknown"),
            "num_hidden_layers": config.hf_config.text_config.num_hidden_layers,
            "num_key_value_heads": config.hf_config.text_config.num_key_value_heads,
            "dynamic_sparse_threshold": self.dynamic_sparse_threshold,
            "video_gating_threshold": self.video_gating_threshold,
            "page_size": self.page_size,
            "video_len": 0,
            "num_video_pages": 0,
            "steps_info": {},
        }

    # ------------------------------------------------------------------
    # Step lifecycle (mirrors AttentionWeightCollector)
    # ------------------------------------------------------------------

    def start_step(self):
        if not self.enabled:
            return
        self.current_step_data = {}

    def add_layer(self, layer_idx: int, metrics: dict):
        """Record sparse metrics for one layer.

        Called from _load_layer_quest after page selection.

        Args:
            layer_idx: transformer layer index
            metrics: dict with keys:
                text_scores_sorted: [T] sorted text Q·K scores (descending)
                page_scores:        [P] Quest upper-bound per page
                page_cutoff:        scalar – dynamic_sparse_threshold percentile
                gate_cutoff:        scalar or None – video_gating_threshold percentile
                global_score:       scalar or None – global video upper bound
                selected_mask:      [P] bool – which pages kept
                video_tokens_used:  int
                video_len:          int
        """
        if not self.enabled:
            return
        # Move tensors to CPU to avoid accumulating GPU memory.
        cpu_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                cpu_metrics[k] = v.cpu()
            else:
                cpu_metrics[k] = v
        self.current_step_data[layer_idx] = cpu_metrics

    def finish_step(self):
        """Save per-step data to disk."""
        if not self.enabled:
            return
        num_layers = len(self.current_step_data)
        if num_layers == 0:
            return

        # Collect per-layer scalar fields for metadata
        layer_keys = sorted(self.current_step_data.keys())
        self.metadata["steps_info"][str(self.steps_completed)] = {
            "video_tokens_used": {
                str(k): self.current_step_data[k]["video_tokens_used"]
                for k in layer_keys
            },
            "num_pages_kept": {
                str(k): (
                    self.current_step_data[k]["selected_mask"].sum().item()
                    if self.current_step_data[k]["selected_mask"] is not None
                    else self.metadata.get("num_video_pages", 0)
                )
                for k in layer_keys
            },
        }

        save_path = self.metrics_dir / f"step_{self.steps_completed:04d}.pt"
        torch.save(self.current_step_data, save_path)

        self.steps_completed += 1
        self.current_step_data = {}

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def set_video_info(self, video_len: int, num_video_pages: int):
        if not self.enabled:
            return
        self.metadata["video_len"] = video_len
        self.metadata["num_video_pages"] = num_video_pages

    def save_metadata(self):
        if not self.enabled:
            return
        self.metadata["num_steps"] = self.steps_completed
        meta_path = self.metrics_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Sparse metrics saved to {self.metrics_dir} ({self.steps_completed} steps)")

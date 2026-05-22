#!/usr/bin/env python3
"""Visualize attention weights collected during FlexCacheGen decoding.

Generates a step-index × token-index heatmap for a given (layer, head) pair.
Format follows the reference style:
  - Wide aspect ratio
  - X-axis: ``token index``, left to right
  - Y-axis: ``step index``, inverted (step 0 at top)
  - Colormap: ``Reds`` (white → dark red), vmin=0, vmax=1
  - Background spans: alternating light-purple / light-green bands for
    video-frame / text regions
  - Colorbar on the right, label ``Value``, ticks 0.0–1.0

Usage:
    python scripts/visualize_attention.py --run_dir <path> --layer 0 --head 21
    python scripts/visualize_attention.py --run_dir <path> --layer 0 --head 21 --step-range 0:30
"""

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Colour palette (from reference specification)
# ---------------------------------------------------------------------------
BG_VIDEO = "#EAF4EA"  # light green (matching a.png green blocks)
BG_TEXT = "#EAEAFB"   # light blue-purple (matching a.png blue-purple blocks)


# ---------------------------------------------------------------------------
# Data loading with in-memory cache
# ---------------------------------------------------------------------------


class StepCache:
    """Load and cache per-step attention weight tensors.

    Each step file is [num_layers, num_heads, seq_len]. Loading all steps from
    disk once and slicing per (layer, head) eliminates redundant I/O when
    plotting multiple layers or heads (e.g. --all-heads mode).
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.weights_dir = Path(run_dir) / "attn_weights"
        self._cache: dict[int, torch.Tensor] = {}  # step_idx -> Tensor on CPU

    def load_step(self, step_idx: int) -> torch.Tensor:
        """Return tensor [num_layers, num_heads, seq_len] (cached)."""
        if step_idx not in self._cache:
            path = self.weights_dir / f"step_{step_idx:04d}.pt"
            self._cache[step_idx] = torch.load(path, weights_only=True, map_location="cpu")
        return self._cache[step_idx]

    def slice_layer_head(self, step_idx: int, layer: int, head: int) -> np.ndarray:
        """Return 1-D float32 numpy array of attention weights for (layer, head)."""
        t = self.load_step(step_idx)
        return t[layer, head].numpy().astype(np.float32)


def load_metadata(run_dir: str) -> dict:
    with open(Path(run_dir) / "attn_weights" / "metadata.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _build_data_matrix(
    cache: StepCache,
    layer: int,
    head: int,
    step_indices: list[int],
    token_range: tuple[int, int] | None,
) -> tuple[np.ndarray, int]:
    """Extract and pad attention weights into a [n_steps, max_seq_len] matrix.

    Returns (data_matrix, token_offset).
    """
    rows = [cache.slice_layer_head(s, layer, head) for s in step_indices]
    max_len = max(len(r) for r in rows)
    n_steps = len(rows)

    # Pre-allocate with zeros (avoid NaN→0 conversion pass)
    data = np.zeros((n_steps, max_len), dtype=np.float32)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r

    if token_range is not None:
        t0, t1 = token_range
        t1 = min(t1, max_len)
        data = data[:, t0:t1]
        return data, t0
    return data, 0


def plot_attention_heatmap(
    run_dir: str,
    layer: int,
    head: int,
    step_range: tuple[int, int] | None = None,
    token_range: tuple[int, int] | None = None,
    output: str | None = None,
    dpi: int = 150,
    cache: StepCache | None = None,
    vmax: float | None = None,
    norm: str = "power",
):
    """Generate the canonical step × token attention heatmap.

    Args:
        run_dir: Path containing ``attn_weights/``.
        layer: Layer index (0-based).
        head: Q-head index (0-based).
        step_range: Optional (first_step, last_step_exclusive).  Defaults to
                    all available steps.
        token_range: Optional (first_token, last_token_exclusive).  Defaults
                     to the full sequence length.
        output: If given, save PNG to this path instead of showing.
        dpi: Output resolution.
        cache: Optional pre-built StepCache (reused across calls).
        vmax: Colormap upper bound. Defaults to the 98th percentile of the
              visible data.
        norm: 'linear' or 'power' (default).  Power-law (gamma=0.4) expands
              low-value contrast for long sequences where raw weights are
              ~1/seq_len ≈ 0.0005.
    """
    meta = load_metadata(run_dir)
    num_steps = meta["num_steps"]
    video_len = meta.get("video_len", 0)
    text_prefix_len = meta.get("text_prefix_len", 0)
    tokens_per_frame = meta.get("tokens_per_frame", 0)

    if cache is None:
        cache = StepCache(run_dir)

    if step_range is None:
        step_range = (0, num_steps)
    step_start, step_end = step_range
    step_end = min(step_end, num_steps)
    step_indices = list(range(step_start, step_end))

    # ------------------------------------------------------------------
    # Build data matrix
    # ------------------------------------------------------------------
    data, token_offset = _build_data_matrix(cache, layer, head, step_indices, token_range)

    # -- auto vmax ---------------------------------------------------------
    if vmax is None:
        flat = data[data > 0]  # exclude zero-padding
        if len(flat) > 0:
            vmax = float(np.percentile(flat, 98))
            vmax = max(vmax, 0.005)  # floor: always show some gradient
        else:
            vmax = 1.0

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    display_steps, display_tokens = data.shape

    # Debug fallback: if metadata is missing video_len / tokens_per_frame,
    # the background spans would never render. Force test defaults.
    if video_len == 0 or tokens_per_frame == 0:
        print("WARNING: metadata missing video_len or tokens_per_frame, "
              "forcing test defaults (video_len=display_tokens, tokens_per_frame=250)")
        video_len = display_tokens
        tokens_per_frame = 250

    fig_w = max(12, min(24, display_tokens / 120))
    fig_h = max(4, min(12, display_steps / 15))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # -- background spans (drawn FIRST, fully opaque, below heatmap) --------
    if video_len > 0:
        vis_start = 0
        vis_end = display_tokens

        def _draw_span_bg(x0, x1, color):
            x0 = max(x0, vis_start)
            x1 = min(x1, vis_end)
            if x1 > x0:
                # Full-height background fill
                ax.axvspan(x0 - 0.5, x1 - 0.5, facecolor=color, alpha=1.0,
                           zorder=0, linewidth=0)

        # Draw alternating video frames using ABSOLUTE token indices
        if tokens_per_frame > 0:
            v0_abs = 0
            fi = 0
            while v0_abs < video_len:
                v1_abs = min(v0_abs + tokens_per_frame, video_len)
                x0_plot = v0_abs - token_offset
                x1_plot = v1_abs - token_offset
                if x1_plot > vis_start and x0_plot < vis_end:
                    color = BG_VIDEO if fi % 2 == 0 else BG_TEXT
                    _draw_span_bg(x0_plot, x1_plot, color)
                v0_abs = v1_abs
                fi += 1
        else:
            _draw_span_bg(0 - token_offset, video_len - token_offset, BG_VIDEO)

        # Draw text region
        _draw_span_bg(video_len - token_offset, display_tokens, BG_TEXT)

    # -- heatmap (hard-threshold alpha: sharp signal, clean background) -----
    if norm == "power":
        normalized = data ** 0.4  # PowerNorm(gamma=0.4) equivalent
        normalized = normalized / (vmax ** 0.4)
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        normalized = np.clip(data / vmax, 0.0, 1.0)

    cmap = plt.cm.Reds
    rgba = cmap(normalized)                 # [H, W, 4] RGBA
    # Hard threshold: values < 2% → fully transparent → clean background
    # values >= 2%  → boosted opacity (min 0.6) → sharp red signal
    alpha_mask = np.where(
        normalized < 0.02,
        0.0,
        np.clip(normalized + 0.5, 0.6, 1.0),
    )
    rgba[:, :, 3] = alpha_mask

    im = ax.imshow(
        rgba,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        zorder=5,
    )

    # -- frame / region boundary lines (on top) -----------------------------
    if video_len > 0:
        if tokens_per_frame > 0:
            v1_abs = tokens_per_frame
            while v1_abs < video_len:
                x_plot = v1_abs - token_offset
                if 0 <= x_plot <= display_tokens:
                    ax.axvline(x=x_plot - 0.5, color="gray", linewidth=0.8,
                               linestyle=":", alpha=0.8, zorder=15)
                v1_abs += tokens_per_frame
        # Main boundary between video and text
        v_end_plot = video_len - token_offset
        if 0 <= v_end_plot <= display_tokens:
            ax.axvline(x=v_end_plot - 0.5, color="black", linewidth=1.0,
                       linestyle="--", alpha=0.8, zorder=15)

    # -- axes -------------------------------------------------------------
    ax.set_xlabel("token index")
    ax.set_ylabel("step index")

    y_ticks = _nice_ticks(step_start, step_end - 1, target=8)
    ax.set_yticks([y - step_start for y in y_ticks])
    ax.set_yticklabels([str(y) for y in y_ticks])

    x_ticks = _nice_ticks(token_offset, token_offset + display_tokens - 1, target=8)
    ax.set_xticks([x - token_offset for x in x_ticks])
    ax.set_xticklabels([str(x) for x in x_ticks])

    # -- colorbar ---------------------------------------------------------
    if norm == "power":
        cb_norm = mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
    else:
        cb_norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=cb_norm, cmap=plt.cm.Reds)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Value", rotation=270, labelpad=12)
    cbar_ticks = np.linspace(0.0, vmax, 6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t:.3f}" if t < 0.01 else f"{t:.2f}" for t in cbar_ticks])

    # -- title ------------------------------------------------------------
    model = meta.get("model_type", "")
    ax.set_title(f"{model}  —  layer {layer}  head {head}")

    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved {output}")
    else:
        plt.show()
    plt.close(fig)


def plot_batch(
    run_dir: str,
    pairs: list[tuple[int, int]],
    step_range: tuple[int, int] | None = None,
    token_range: tuple[int, int] | None = None,
    output_dir: str | None = None,
    dpi: int = 150,
    vmax: float | None = None,
    norm: str = "power",
):
    """Plot multiple (layer, head) pairs, reusing cached step data."""
    import os

    if output_dir is None:
        output_dir = os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    cache = StepCache(run_dir)

    for layer, head in pairs:
        out = os.path.join(output_dir, f"L{layer:02d}_H{head:02d}.png")
        print(f"  Plotting layer {layer} head {head}  →  {out}")
        plot_attention_heatmap(
            run_dir=run_dir,
            layer=layer,
            head=head,
            step_range=step_range,
            token_range=token_range,
            output=out,
            dpi=dpi,
            cache=cache,
            vmax=vmax,
            norm=norm,
        )

    print(f"Done — {len(pairs)} plot(s) saved to {output_dir}")


def _nice_ticks(lo: int, hi: int, target: int = 8) -> list[int]:
    """Return a list of nice integer tick positions covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / (target - 1)
    for nice in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if raw_step <= nice:
            break
    ticks = [lo // nice * nice]
    while ticks[-1] < hi:
        ticks.append(ticks[-1] + nice)
    if ticks[-1] < hi:
        ticks.append(ticks[-1] + nice)
    return [t for t in ticks if lo <= t <= hi]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention weights (step × token heatmap)"
    )
    parser.add_argument("--run_dir", required=True, help="Run output directory")
    parser.add_argument("--layer", type=int, required=True, help="Layer index (0-based)")
    parser.add_argument("--head", type=int, required=True, help="Q-head index (0-based)")
    parser.add_argument(
        "--step-range", default=None,
        help="Step range, e.g. '0:30' (start:end, end exclusive)",
    )
    parser.add_argument(
        "--token-range", default=None,
        help="Token range, e.g. '0:1500' (start:end, end exclusive)",
    )
    parser.add_argument("--output", default=None, help="Save plot to file")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--vmax", type=float, default=None,
                        help="Colormap upper bound (default: auto 98th pct)")
    parser.add_argument("--norm", choices=["linear", "power"], default="power",
                        help="Colormap normalisation (default: power, gamma=0.4)")

    args = parser.parse_args()

    step_range = None
    if args.step_range is not None:
        parts = args.step_range.split(":")
        step_range = (int(parts[0]), int(parts[1]) if len(parts) > 1 else None)

    token_range = None
    if args.token_range is not None:
        parts = args.token_range.split(":")
        token_range = (int(parts[0]), int(parts[1]) if len(parts) > 1 else None)

    meta = load_metadata(args.run_dir)
    if args.layer < 0 or args.layer >= meta["num_hidden_layers"]:
        print(f"Layer {args.layer} out of range [0, {meta['num_hidden_layers'] - 1}]")
        return
    if args.head < 0 or args.head >= meta["num_attention_heads"]:
        print(f"Head {args.head} out of range [0, {meta['num_attention_heads'] - 1}]")
        return

    if step_range is not None and step_range[1] is None:
        step_range = (step_range[0], meta["num_steps"])

    plot_attention_heatmap(
        run_dir=args.run_dir,
        layer=args.layer,
        head=args.head,
        step_range=step_range,
        token_range=token_range,
        output=args.output,
        dpi=args.dpi,
        vmax=args.vmax,
        norm=args.norm,
    )


if __name__ == "__main__":
    main()

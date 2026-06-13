#!/usr/bin/env python3
"""
Run VLM inference with attention weight collection, then generate plots.

Generates step-index × token-index heatmaps matching the reference style:
  - X-axis: ``token index``, Y-axis: ``step index`` (inverted)
  - Colormap: ``Reds`` (white → dark red), vmin=0, vmax=1
  - Background spans: light-purple / light-green bands for text / video regions
  - Colorbar on the right

Usage:
    # Inference + plot one (layer, head)
    python analyze_attention.py --layer 0 --head 21
    python analyze_attention.py --video ./test/video/28s.mp4 --layer 0 --head 0

    # Plot every head in a layer
    python analyze_attention.py --layer 5 --all-heads

    # Plot every layer for one head
    python analyze_attention.py --head 10 --all-layers

    # Only inference (no plot)
    python analyze_attention.py --skip-plot

    # Only plot from an existing run
    python analyze_attention.py --run-dir <path> --skip-inference --layer 0 --head 0
    path like: /nvme1n1p1/lyc/flexcachegen_outputs/20260522_063551_28s

    # Focus on a step / token sub-range
    python analyze_attention.py --layer 0 --head 0 --step-range 0:30 --token-range 0:1500
"""

import argparse
import importlib.util
import os
import sys

from flexcachegen.config import OUTPUT_ROOT
from flexcachegen.engine import VLMEngine

# Import visualize_attention from scripts/ by absolute path
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
_va_path = os.path.join(_scripts_dir, "visualize_attention.py")
_spec = importlib.util.spec_from_file_location("visualize_attention", _va_path)
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)

load_metadata = _va.load_metadata
plot_batch = _va.plot_batch

DATASET_ROOT = os.environ.get("DATASET_ROOT", "/data/lyc/datasets")


def find_latest_run_dir():
    """Find the most recently created run directory under OUTPUT_ROOT."""
    root = os.path.join(OUTPUT_ROOT)
    if not os.path.isdir(root):
        return None
    dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def generate_plots(
    run_dir: str,
    layer: int | None = None,
    head: int | None = None,
    all_layers: bool = False,
    all_heads: bool = False,
    step_range: str | None = None,
    token_range: str | None = None,
    dpi: int = 150,
    vmax: float | None = None,
    norm: str = "power",
):
    """Generate step×token attention heatmaps, reusing cached step data.

    Args:
        run_dir: Path to the run output directory (contains attn_weights/).
        layer: Single layer index.
        head: Single head index.
        all_layers: If True, plot every layer (head must be a single index).
        all_heads: If True, plot every head (layer must be a single index).
        step_range: e.g. '0:30' (start:end, end exclusive).
        token_range: e.g. '0:1500' (start:end, end exclusive).
        dpi: Figure DPI.
    """
    meta = load_metadata(run_dir)
    n_layers = meta["num_hidden_layers"]
    n_heads = meta["num_attention_heads"]

    _sr = None
    if step_range is not None:
        parts = step_range.split(":")
        _sr = (int(parts[0]), int(parts[1]) if len(parts) > 1 else None)

    _tr = None
    if token_range is not None:
        parts = token_range.split(":")
        _tr = (int(parts[0]), int(parts[1]) if len(parts) > 1 else None)

    # Build (layer, head) pairs
    pairs: list[tuple[int, int]] = []
    if all_layers and all_heads:
        pairs = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    elif all_layers:
        h = head if head is not None else 0
        pairs = [(l, h) for l in range(n_layers)]
    elif all_heads:
        l = layer if layer is not None else 0
        pairs = [(l, h) for h in range(n_heads)]
    else:
        l = layer if layer is not None else 0
        h = head if head is not None else 0
        pairs = [(l, h)]

    plots_dir = os.path.join(run_dir, "plots")

    # Single shared cache: each step file loaded from disk only once
    plot_batch(
        run_dir=run_dir,
        pairs=pairs,
        step_range=_sr,
        token_range=_tr,
        output_dir=plots_dir,
        dpi=dpi,
        vmax=vmax,
        norm=norm,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM inference with attention weight analysis"
    )
    # -- Inference ----------------------------------------------------------
    parser.add_argument(
        "--video", default="./test/video/28s.mp4", help="Path to input video"
    )
    parser.add_argument(
        "--question", default="Please describe this video in detail.",
        help="Question about the video",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--no-info", action="store_true",
        help="Use generate_single (no profiling) instead of generate_single_info",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Skip inference, plot from existing run directory",
    )
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="Only plot from --run-dir, don't run inference",
    )
    parser.add_argument(
        "--skip-plot", action="store_true",
        help="Only run inference, don't generate plots",
    )
    parser.add_argument(
        "--save-sparse-metrics", action="store_true",
        help="Collect sparse-pruning metrics (Quest page selection, gating) during decode",
    )

    # -- Plot ---------------------------------------------------------------
    parser.add_argument("--layer", type=int, default=None, help="Layer index to plot")
    parser.add_argument("--head", type=int, default=None, help="Q-head index to plot")
    parser.add_argument("--all-layers", action="store_true",
                        help="Plot all layers for the given --head")
    parser.add_argument("--all-heads", action="store_true",
                        help="Plot all heads for the given --layer")
    parser.add_argument("--step-range", default=None,
                        help="Step range e.g. '0:30' (start:end, end exclusive)")
    parser.add_argument("--token-range", default=None,
                        help="Token range e.g. '0:1500' (start:end, end exclusive)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--vmax", type=float, default=None,
                        help="Colormap upper bound (default: auto 98th pct of data)")
    parser.add_argument("--norm", choices=["linear", "power"], default="power",
                        help="Colormap normalisation (default: power, gamma=0.4)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Inference
    # ------------------------------------------------------------------
    run_dir = args.run_dir

    if not args.skip_inference:
        print("=" * 60)
        print(" Stage 1: Inference with attention weight collection")
        print("=" * 60)

        vlm = VLMEngine(
            model_type="qwen3vl-8b",
            save_attention_weights=True,
            save_sparse_metrics=args.save_sparse_metrics,
            max_new_tokens=args.max_new_tokens,
        )

        if args.no_info:
            output = vlm.generate_single(args.video, args.question)
        else:
            output = vlm.generate_single_info(args.video, args.question)

        run_dir = find_latest_run_dir()
        print(f"Output text: {output[:200]}...")

    # ------------------------------------------------------------------
    # 2. Plot
    # ------------------------------------------------------------------
    if args.skip_plot:
        print(f"Run directory: {run_dir}")
        print("Skipping plot generation (--skip-plot).")
        return

    if run_dir is None:
        print("ERROR: No run directory found. Run inference first or use --run-dir.")
        sys.exit(1)

    attn_dir = os.path.join(run_dir, "attn_weights")
    if not os.path.isdir(attn_dir):
        print(f"ERROR: No attn_weights/ found in {run_dir}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(" Stage 2: Plot generation")
    print(f" Run dir: {run_dir}")
    print("=" * 60)

    generate_plots(
        run_dir=run_dir,
        layer=args.layer,
        head=args.head,
        all_layers=args.all_layers,
        all_heads=args.all_heads,
        step_range=args.step_range,
        token_range=args.token_range,
        dpi=args.dpi,
        vmax=args.vmax,
        norm=args.norm,
    )


if __name__ == "__main__":
    main()

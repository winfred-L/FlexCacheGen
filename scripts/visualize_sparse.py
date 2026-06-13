#!/usr/bin/env python3
"""
Visualize sparse-pruning metrics collected during decode.

Generates three plot types from SparseMetricsCollector output:
  1. Text Q·K distribution + dual cutoffs + page scores (per-step, per-layer)
  2. Page selection heatmap (decode steps × video pages)
  3. Global gating timeline + sparsity-by-layer summary

Usage:
    # Plot a specific step and layer
    python scripts/visualize_sparse.py --run-dir <path> --step 10 --layer 5

    # Plot the gating timeline for a layer
    python scripts/visualize_sparse.py --run-dir <path> --layer 5 --mode timeline

    # Plot page selection heatmap for a layer
    python scripts/visualize_sparse.py --run-dir <path> --layer 5 --mode heatmap

    # Plot per-layer sparsity summary
    python scripts/visualize_sparse.py --run-dir <path> --mode summary

Run dir layout:
    <run_dir>/
      sparse_metrics/
        metadata.json
        step_0000.pt
        step_0001.pt
        ...
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_run(run_dir: str) -> tuple[dict, list[dict]]:
    """Load sparse metrics from a run directory.

    Returns (metadata, steps_data) where steps_data[step] is a dict
    {str(layer_idx): metrics_dict}.
    """
    metrics_dir = Path(run_dir) / "sparse_metrics"
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Sparse metrics directory not found: {metrics_dir}")

    with open(metrics_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_steps = metadata["num_steps"]
    steps_data = []
    for step in range(num_steps):
        path = metrics_dir / f"step_{step:04d}.pt"
        steps_data.append(torch.load(path, map_location="cpu", weights_only=False))

    return metadata, steps_data


def _fig_save(fig, save_dir: str, name: str, dpi: int = 150):
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1: Text Q·K distribution + cutoffs + page scores
# ---------------------------------------------------------------------------

def plot_text_distribution(
    metadata: dict,
    steps_data: list[dict],
    step: int,
    layer: int,
    save_dir: str,
):
    """Histogram of text Q·K scores with cutoff lines and page score overlay.

    X-axis = Q·K score (raw dot product sum). Y-axis = density.
    Vertical lines = page_cutoff (dynamic_sparse_threshold), gate_cutoff.
    Scatter = per-page Quest scores (red=kept, gray=pruned).
    """
    step_data = steps_data[step]
    lkey = str(layer)
    if lkey not in step_data:
        print(f"  Layer {layer} not found in step {step}")
        return
    m = step_data[lkey]

    text_scores = m["text_scores_sorted"].numpy()  # descending
    page_scores = m["page_scores"].numpy() if m["page_scores"] is not None else None
    page_cutoff = m["page_cutoff"]
    gate_cutoff = m["gate_cutoff"]
    selected_mask = m["selected_mask"].numpy() if m["selected_mask"] is not None else None
    gated = m.get("gated", False)

    dyn_thresh = metadata["dynamic_sparse_threshold"]
    gate_thresh = metadata.get("video_gating_threshold")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Text score histogram
    ax.hist(text_scores, bins=min(80, len(text_scores) // 10),
            color="#5B9BD5", edgecolor="white", alpha=0.75, density=True,
            label="text Q·K scores")

    # Cutoff lines
    ylim = ax.get_ylim()
    if page_cutoff is not None:
        ax.axvline(page_cutoff, color="#E74C3C", linewidth=2, linestyle="--",
                   label=f"page cutoff (p={dyn_thresh}, val={page_cutoff:.1f})")
    if gate_cutoff is not None:
        ax.axvline(gate_cutoff, color="#8E44AD", linewidth=2, linestyle=":",
                   label=f"gate cutoff (p={gate_thresh}, val={gate_cutoff:.1f})")
    ax.set_ylim(ylim)

    # Page scores as scatter on a second y-axis
    if page_scores is not None and len(page_scores) > 0:
        ax2 = ax.twiny()
        y_jitter = np.full(len(page_scores), ax.get_ylim()[1] * 0.92)
        kept = selected_mask if selected_mask is not None else np.ones(len(page_scores), dtype=bool)
        pruned = ~kept
        ax2.scatter(page_scores[kept], y_jitter[kept], c="#E74C3C", s=30, marker="|",
                    linewidths=1.5, label=f"kept pages ({kept.sum()})", zorder=5)
        if pruned.sum() > 0:
            ax2.scatter(page_scores[pruned], y_jitter[pruned], c="#BDC3C7", s=20, marker="|",
                        linewidths=1, label=f"pruned pages ({pruned.sum()})", zorder=4)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([])
        ax2.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Q·K score (Σ_h Σ_d q·k)")
    ax.set_ylabel("Density")
    title = (f"Step {step}  Layer {layer}  "
             f"video_tokens={m['video_tokens_used']}/{m['video_len']}"
             f"{' [GATED]' if gated else ''}")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    _fig_save(fig, save_dir, f"dist_step{step:04d}_layer{layer:02d}")


# ---------------------------------------------------------------------------
# Plot 2: Page selection heatmap
# ---------------------------------------------------------------------------

def plot_page_heatmap(
    metadata: dict,
    steps_data: list[dict],
    layer: int,
    save_dir: str,
    step_range: tuple[int, int] | None = None,
):
    """2D heatmap: rows = decode steps, columns = video pages.

    Cell color = page score (normalised per row). Overlaid hatch or border
    indicates pruned pages.
    """
    num_steps = len(steps_data)
    if step_range:
        s0, s1 = step_range
        s1 = min(s1, num_steps)
    else:
        s0, s1 = 0, num_steps
    num_plot_steps = s1 - s0

    lkey = str(layer)
    num_pages = metadata["num_video_pages"]
    if num_pages == 0:
        print("  No video pages to plot")
        return

    # Collect per-step data
    score_matrix = np.full((num_plot_steps, num_pages), np.nan)
    kept_matrix = np.zeros((num_plot_steps, num_pages), dtype=bool)
    gated_steps = np.zeros(num_plot_steps, dtype=bool)
    video_used = np.zeros(num_plot_steps, dtype=int)

    for si in range(s0, s1):
        m = steps_data[si].get(lkey, {})
        row = si - s0
        video_used[row] = m.get("video_tokens_used", 0)
        gated_steps[row] = m.get("gated", False)
        ps = m.get("page_scores")
        sm = m.get("selected_mask")
        if ps is not None:
            ps_np = ps.numpy()
            score_matrix[row, :len(ps_np)] = ps_np
        if sm is not None:
            sm_np = sm.numpy()
            kept_matrix[row, :len(sm_np)] = sm_np
        else:
            # All pages kept
            kept_matrix[row, :] = True

    # Normalise scores per row for colormap
    score_viz = score_matrix.copy()
    for row in range(num_plot_steps):
        row_scores = score_viz[row]
        valid = ~np.isnan(row_scores)
        if valid.sum() > 1:
            vmin = np.nanmin(row_scores[valid])
            vmax = np.nanmax(row_scores[valid])
            if vmax > vmin:
                score_viz[row, valid] = (row_scores[valid] - vmin) / (vmax - vmin + 1e-8)

    fig, ax = plt.subplots(figsize=(14, max(5, num_plot_steps * 0.25)))

    cmap = matplotlib.colormaps["RdYlGn"]
    im = ax.imshow(score_viz, aspect="auto", origin="upper", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Mark pruned pages with a dot or lighter overlay
    for row in range(num_plot_steps):
        pruned_cols = np.where(~kept_matrix[row])[0]
        if len(pruned_cols) > 0:
            ax.scatter(pruned_cols, np.full_like(pruned_cols, row, dtype=float),
                       marker="x", color="black", s=4, alpha=0.5, linewidths=0.3)

    # Mark gated steps on the right
    for row in range(num_plot_steps):
        if gated_steps[row]:
            ax.annotate("G", (num_pages + 0.5, row), fontsize=6, color="red",
                        ha="center", va="center")

    ax.set_xlabel("Video page index")
    ax.set_ylabel(f"Decode step ({s0}–{s1 - 1})")
    ax.set_title(f"Page selection heatmap — Layer {layer}")
    ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Page score (row-normalised)")

    _fig_save(fig, save_dir, f"heatmap_layer{layer:02d}")


# ---------------------------------------------------------------------------
# Plot 3: Global gating timeline + per-layer sparsity summary
# ---------------------------------------------------------------------------

def plot_gating_timeline(
    metadata: dict,
    steps_data: list[dict],
    layer: int,
    save_dir: str,
):
    """Line plot: global_score vs gate_cutoff over decode steps.

    Also shows video_tokens_used / video_len ratio as a background fill.
    """
    lkey = str(layer)
    num_steps = len(steps_data)

    steps_x = []
    global_scores = []
    gate_cutoffs = []
    page_cutoffs = []
    ratios = []
    gated_flags = []

    for si in range(num_steps):
        m = steps_data[si].get(lkey, {})
        if m.get("global_score") is None and m.get("page_cutoff") is None:
            continue
        steps_x.append(si)
        if m.get("global_score") is not None:
            global_scores.append(m["global_score"])
            gate_cutoffs.append(m.get("gate_cutoff", np.nan))
        else:
            global_scores.append(np.nan)
            gate_cutoffs.append(np.nan)
        if m.get("page_cutoff") is not None:
            page_cutoffs.append(m["page_cutoff"])
        else:
            page_cutoffs.append(np.nan)
        vlen = m.get("video_len", 1)
        ratios.append(m.get("video_tokens_used", 0) / max(vlen, 1))
        gated_flags.append(m.get("gated", False))

    if len(steps_x) == 0:
        print(f"  No gating data for layer {layer}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: scores and cutoffs
    ax1.plot(steps_x, global_scores, color="#3498DB", linewidth=1.5, marker=".",
             markersize=3, label="global_score")
    ax1.plot(steps_x, gate_cutoffs, color="#8E44AD", linewidth=1.5, linestyle="--",
             label=f"gate_cutoff (p={metadata.get('video_gating_threshold')})")
    ax1.plot(steps_x, page_cutoffs, color="#E74C3C", linewidth=1.2, linestyle=":",
             label=f"page_cutoff (p={metadata.get('dynamic_sparse_threshold')})")

    # Highlight gated steps
    gated_x = [steps_x[i] for i, g in enumerate(gated_flags) if g]
    if gated_x:
        for gx in gated_x:
            ax1.axvline(gx, color="red", alpha=0.15, linewidth=0.5)

    ax1.set_ylabel("Score")
    ax1.set_title(f"Gating timeline — Layer {layer}")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: video ratio
    ax2.fill_between(steps_x, ratios, color="#2ECC71", alpha=0.4, step="mid")
    ax2.plot(steps_x, ratios, color="#27AE60", linewidth=1)
    # Mark gated steps
    for gx in gated_x:
        ax2.axvline(gx, color="red", alpha=0.15, linewidth=0.5)
    ax2.set_xlabel("Decode step")
    ax2.set_ylabel("Video ratio\n(video_tokens_used / video_len)")
    ax2.set_ylim(-0.05, 1.15)
    ax2.grid(True, alpha=0.3)
    if gated_x:
        ax2.text(0.99, 0.95, f"Gated: {len(gated_x)}/{len(steps_x)} steps",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, color="red", fontweight="bold")

    _fig_save(fig, save_dir, f"timeline_layer{layer:02d}")


# ---------------------------------------------------------------------------
# Plot 4: Per-layer sparsity summary
# ---------------------------------------------------------------------------

def plot_layer_summary(
    metadata: dict,
    steps_data: list[dict],
    save_dir: str,
):
    """Bar chart: average fraction of pages kept per layer across all decode steps.

    Also shows how many steps were gated (zero video) per layer.
    """
    num_layers = metadata["num_hidden_layers"]
    num_steps = metadata["num_steps"]
    num_pages = metadata.get("num_video_pages", 0)
    if num_pages == 0:
        print("  No video pages — summary not applicable")
        return

    avg_kept = np.zeros(num_layers)
    avg_ratio = np.zeros(num_layers)
    gated_counts = np.zeros(num_layers, dtype=int)

    for layer in range(num_layers):
        lkey = str(layer)
        total_kept = 0
        total_ratio = 0.0
        valid_steps = 0
        for si in range(num_steps):
            m = steps_data[si].get(lkey, {})
            if not m:
                continue
            valid_steps += 1
            sm = m.get("selected_mask")
            if sm is not None:
                total_kept += sm.sum().item()
            else:
                total_kept += num_pages
            vlen = m.get("video_len", 1)
            total_ratio += m.get("video_tokens_used", 0) / max(vlen, 1)
            if m.get("gated", False):
                gated_counts[layer] += 1

        if valid_steps > 0:
            avg_kept[layer] = total_kept / (valid_steps * num_pages)
            avg_ratio[layer] = total_ratio / valid_steps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: average fraction of pages kept
    colors = ["#E74C3C" if r < 0.3 else "#F39C12" if r < 0.6 else "#2ECC71"
              for r in avg_kept]
    bars = ax1.bar(range(num_layers), avg_kept, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=metadata.get("dynamic_sparse_threshold", 0.1), color="#3498DB",
                linestyle="--", linewidth=1.5,
                label=f"dynamic_sparse_threshold={metadata['dynamic_sparse_threshold']}")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Avg fraction of pages kept")
    ax1.set_title("Per-layer sparsity summary")
    ax1.set_xticks(range(0, num_layers, max(1, num_layers // 18)))
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    # Annotate bar values for extremes
    for i, (r, c) in enumerate(zip(avg_kept, colors)):
        if r < 0.2 or r > 0.8:
            ax1.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=6, color=c)

    # Bottom: gated steps per layer
    ax2.bar(range(num_layers), gated_counts, color="#E74C3C", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Gated steps")
    ax2.set_title(f"Video gating frequency (total {num_steps} steps)")
    ax2.set_xticks(range(0, num_layers, max(1, num_layers // 18)))
    ax2.grid(True, axis="y", alpha=0.3)

    _fig_save(fig, save_dir, "summary_layers")


# ---------------------------------------------------------------------------
# Plot 5: All-in-one per-step per-layer dashboard
# ---------------------------------------------------------------------------

def plot_step_dashboard(
    metadata: dict,
    steps_data: list[dict],
    step: int,
    save_dir: str,
):
    """Multi-panel figure showing all layers for a single decode step."""
    num_layers = metadata["num_hidden_layers"]
    step_data = steps_data[step]

    # Collect per-layer data
    layers_with_data = []
    for layer in range(num_layers):
        lkey = str(layer)
        if lkey in step_data:
            layers_with_data.append(layer)

    if not layers_with_data:
        print(f"  No data for step {step}")
        return

    ncols = min(6, len(layers_with_data))
    nrows = math.ceil(len(layers_with_data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
                              squeeze=False)

    dyn_thresh = metadata["dynamic_sparse_threshold"]
    gate_thresh = metadata.get("video_gating_threshold")

    for pi, layer in enumerate(layers_with_data):
        ax = axes[pi // ncols][pi % ncols]
        m = step_data[str(layer)]
        text_scores = m["text_scores_sorted"].numpy()
        page_scores = m["page_scores"]
        page_cutoff = m["page_cutoff"]
        gate_cutoff = m["gate_cutoff"]
        gated = m.get("gated", False)
        video_len = m.get("video_len", 0)
        video_used = m.get("video_tokens_used", 0)

        # Compact histogram
        ax.hist(text_scores, bins=30, color="#5B9BD5", alpha=0.6, density=True)

        if page_cutoff is not None:
            ax.axvline(page_cutoff, color="#E74C3C", linewidth=1.2, linestyle="--")
        if gate_cutoff is not None:
            ax.axvline(gate_cutoff, color="#8E44AD", linewidth=1.2, linestyle=":")

        if page_scores is not None:
            ps = page_scores.numpy()
            sm = m.get("selected_mask")
            kept = sm.numpy() if sm is not None else np.ones(len(ps), dtype=bool)
            yh = ax.get_ylim()[1] * 0.9
            ax.scatter(ps[kept], np.full(kept.sum(), yh), c="#E74C3C", s=8, marker="|")
            if (~kept).sum() > 0:
                ax.scatter(ps[~kept], np.full((~kept).sum(), yh), c="#BDC3C7", s=5, marker="|")

        status = "GATED" if gated else f"{video_used}/{video_len}"
        n_kept = m.get("selected_mask").sum().item() if m.get("selected_mask") is not None else "all"
        ax.set_title(f"L{layer} {status} kept={n_kept}", fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for pi in range(len(layers_with_data), nrows * ncols):
        axes[pi // ncols][pi % ncols].set_visible(False)

    fig.suptitle(f"Step {step} — all layers  "
                 f"(dyn={dyn_thresh}, gate={gate_thresh})",
                 fontsize=10)
    fig.tight_layout()
    _fig_save(fig, save_dir, f"dashboard_step{step:04d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize sparse-pruning metrics")
    parser.add_argument("--run-dir", required=True,
                        help="Path to the run output directory (contains sparse_metrics/)")
    parser.add_argument("--step", type=int, default=0,
                        help="Decode step for distribution / dashboard plots")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index for per-layer plots")
    parser.add_argument("--mode", choices=["dist", "heatmap", "timeline", "summary", "dashboard", "all"],
                        default="all",
                        help="Plot type: dist=text Q·K distribution, heatmap=page selection, "
                             "timeline=gating timeline, summary=per-layer bars, "
                             "dashboard=all layers for one step, all=everything")
    parser.add_argument("--all-layers", action="store_true",
                        help="Plot distribution for every layer at the given step")
    parser.add_argument("--all-steps-dashboard", action="store_true",
                        help="Plot dashboard for every decode step")
    args = parser.parse_args()

    metadata, steps_data = _load_run(args.run_dir)
    save_dir = os.path.join(args.run_dir, "sparse_plots")
    os.makedirs(save_dir, exist_ok=True)

    num_steps = metadata["num_steps"]
    num_layers = metadata["num_hidden_layers"]
    print(f"Run: {args.run_dir}")
    print(f"  Steps: {num_steps}, Layers: {num_layers}, "
          f"Video pages: {metadata.get('num_video_pages', 0)}")

    if args.mode in ("dist", "all"):
        if args.all_layers:
            for layer in range(num_layers):
                plot_text_distribution(metadata, steps_data, args.step, layer, save_dir)
        else:
            plot_text_distribution(metadata, steps_data, args.step, args.layer, save_dir)

    if args.mode in ("heatmap", "all"):
        plot_page_heatmap(metadata, steps_data, args.layer, save_dir)

    if args.mode in ("timeline", "all"):
        plot_gating_timeline(metadata, steps_data, args.layer, save_dir)

    if args.mode in ("summary", "all"):
        plot_layer_summary(metadata, steps_data, save_dir)

    if args.mode in ("dashboard", "all"):
        if args.all_steps_dashboard:
            for step in range(num_steps):
                plot_step_dashboard(metadata, steps_data, step, save_dir)
        else:
            plot_step_dashboard(metadata, steps_data, args.step, save_dir)

    print("Done.")


if __name__ == "__main__":
    main()

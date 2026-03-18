"""
Sparse Scatter Triton Kernel for KV Cache Reconstruction
=========================================================

When KV caches are offloaded to CPU with head-level sparsity pruning enabled,
pruned attention heads contain zeros at video token positions. Transferring
these zeros over PCIe is wasteful. Instead, we store compact representations
on CPU and use this Triton kernel to scatter them back into the full layout
on GPU after transfer.

Compact storage layout:
  - Active heads (A heads, not pruned): stored as [B, S, A, D]
    All sequence positions are retained since these heads carry valid data
    everywhere.
  - Pruned heads (P heads): stored as [B, T, P, D]
    Only text token positions (T << S) are stored; video positions are
    known to be zero and are filled directly on GPU, avoiding PCIe transfer.

The kernel reconstructs the full [B, S, H, D] tensor by dispatching one
program per (batch, seq_pos, head) triple. Each program reads from the
appropriate compact buffer or writes zeros, based on two small lookup tables:
  - head_source[H]:      whether each head is active (0) or pruned (1)
  - head_compact_idx[H]: the head's index within its compact buffer
  - text_seq_map[S]:     maps each seq position to its text buffer index,
                         or -1 if the position is a video token

Bandwidth savings example (threshold=0.8, 6/8 heads pruned, 80% video):
  Original transfer : S * 8 * D
  Compact transfer  : S * 2 * D + 0.2*S * 6 * D = 3.2*S*D  (~60% reduction)

Compilation note:
  Only BLOCK_D (= head_dim, typically 128) is a constexpr. All other
  dimensions (S, H, A, P, T) are runtime parameters, so the kernel is
  compiled exactly once regardless of varying sequence lengths or head counts
  across layers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_scatter_kernel(
    # Pointers to input compact buffers
    compact_active_ptr,   # [B, S, A, D] — active heads, all seq positions
    compact_pruned_ptr,   # [B, T, P, D] — pruned heads, text positions only
    # Pointer to output full buffer
    full_ptr,             # [B, S, H, D] — reconstructed full KV cache
    # Metadata pointers (small lookup tables, resident on GPU)
    head_source_ptr,      # [H] int32 — 0=active, 1=pruned
    head_compact_idx_ptr, # [H] int32 — index within the corresponding compact buffer
    text_seq_map_ptr,     # [S] int32 — seq pos → text buffer index (-1 for video)
    # Dimensions — runtime parameters (NOT constexpr to avoid recompilation)
    S,  # sequence length
    H,  # total number of heads (A + P)
    A,  # number of active (non-pruned) heads
    P,  # number of pruned heads
    T,  # number of text token positions
    # Strides for compact_active [B, S, A, D]
    stride_ca_b, stride_ca_s, stride_ca_a, stride_ca_d,
    # Strides for compact_pruned [B, T, P, D]
    stride_cp_b, stride_cp_t, stride_cp_p, stride_cp_d,
    # Strides for full [B, S, H, D]
    stride_f_b, stride_f_s, stride_f_h, stride_f_d,
    # Block size for the head dimension (only constexpr — controls vectorized load width)
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (batch, seq_pos, head) → one D-dimensional vector.
    # Grid is 1D with B*S*H programs total.
    pid = tl.program_id(0)

    # Recover 3D coordinates from flat program ID
    h = pid % H
    s = (pid // H) % S
    b = pid // (H * S)

    # Look up whether this head is active or pruned, and its compact buffer index
    is_pruned = tl.load(head_source_ptr + h)
    compact_idx = tl.load(head_compact_idx_ptr + h)

    # Base offset for the output location full[b, s, h, :]
    out_offset = b * stride_f_b + s * stride_f_s + h * stride_f_h

    # Vectorized index range [0, 1, ..., BLOCK_D-1] for loading/storing D elements
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < BLOCK_D

    if is_pruned == 0:
        # --- Branch 1: Active head ---
        # Data exists at every seq position. Read from compact_active[b, s, compact_idx, :].
        in_offset = b * stride_ca_b + s * stride_ca_s + compact_idx * stride_ca_a
        vals = tl.load(compact_active_ptr + in_offset + d_range * stride_ca_d, mask=d_mask)
        tl.store(full_ptr + out_offset + d_range * stride_f_d, vals, mask=d_mask)
    else:
        # Pruned head — behavior depends on whether this seq position is text or video
        text_idx = tl.load(text_seq_map_ptr + s)

        if text_idx >= 0:
            # --- Branch 2: Pruned head at a text position ---
            # Text positions still have valid data even for pruned heads.
            # Read from compact_pruned[b, text_idx, compact_idx, :].
            # Note: text_idx (not s) is used because the pruned buffer only stores
            # T text positions contiguously, so we need the remapped index.
            in_offset = b * stride_cp_b + text_idx * stride_cp_t + compact_idx * stride_cp_p
            vals = tl.load(compact_pruned_ptr + in_offset + d_range * stride_cp_d, mask=d_mask)
            tl.store(full_ptr + out_offset + d_range * stride_f_d, vals, mask=d_mask)
        else:
            # --- Branch 3: Pruned head at a video position ---
            # This is the data we intentionally skipped transferring from CPU.
            # These values were pruned to zero, so we write zeros directly on GPU.
            zeros = tl.zeros([BLOCK_D], dtype=compact_active_ptr.dtype.element_ty)
            tl.store(full_ptr + out_offset + d_range * stride_f_d, zeros, mask=d_mask)


def sparse_scatter(
    compact_active: torch.Tensor,   # [B, S, A, D]
    compact_pruned: torch.Tensor,   # [B, T, P, D]
    full: torch.Tensor,             # [B, S, H, D]
    head_source: torch.Tensor,      # [H] int32 — 0=active, 1=pruned
    head_compact_idx: torch.Tensor, # [H] int32 — index in compact buffer
    text_seq_map: torch.Tensor,     # [S] int32 — -1 for video positions
):
    """
    Scatter compact active/pruned buffers into the full [B, S, H, D] KV cache layout.

    This is the Python entry point that launches the Triton kernel. It infers all
    dimensions from tensor shapes and passes explicit strides so the kernel is
    agnostic to memory layout (contiguous, transposed, etc.).

    Args:
        compact_active:   KV data for active (non-pruned) heads at all seq positions.
        compact_pruned:   KV data for pruned heads at text-only positions.
                          May be an empty tensor if no heads are pruned.
        full:             Output buffer, pre-allocated on GPU.
        head_source:      Per-head flag indicating active (0) or pruned (1).
        head_compact_idx: Per-head index into the compact_active or compact_pruned buffer.
        text_seq_map:     Per-position mapping to text buffer index; -1 for video positions.
    """
    B, S, H, D = full.shape
    A = compact_active.shape[2]
    P = compact_pruned.shape[2] if compact_pruned.numel() > 0 else 0
    T = compact_pruned.shape[1] if compact_pruned.numel() > 0 else 0

    # BLOCK_D is the only constexpr — since D is fixed (128) across all layers,
    # the kernel is compiled exactly once and reused for every layer/decode step.
    BLOCK_D = triton.next_power_of_2(D)

    # 1D grid: one program per (batch, seq_pos, head) triple
    grid = (B * S * H,)

    _sparse_scatter_kernel[grid](
        compact_active, compact_pruned, full,
        head_source, head_compact_idx, text_seq_map,
        S, H, A, P, T,
        # Explicit strides decouple the kernel from specific memory layouts
        compact_active.stride(0), compact_active.stride(1),
        compact_active.stride(2), compact_active.stride(3),
        compact_pruned.stride(0), compact_pruned.stride(1),
        compact_pruned.stride(2), compact_pruned.stride(3),
        full.stride(0), full.stride(1),
        full.stride(2), full.stride(3),
        BLOCK_D=BLOCK_D,
    )

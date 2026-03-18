import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_scatter_kernel(
    # Pointers to input compact buffers
    compact_active_ptr,   # [B, S, A, D]
    compact_pruned_ptr,   # [B, T, P, D]
    # Pointer to output full buffer
    full_ptr,             # [B, S, H, D]
    # Metadata pointers
    head_source_ptr,      # [H] — 0=active, 1=pruned
    head_compact_idx_ptr, # [H] — index within active or pruned buffer
    text_seq_map_ptr,     # [S] — seq pos → text buffer index (-1 for video)
    # Dimensions
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    A: tl.constexpr,      # number of active heads
    P: tl.constexpr,      # number of pruned heads
    T: tl.constexpr,      # number of text positions
    # Strides for compact_active [B, S, A, D]
    stride_ca_b, stride_ca_s, stride_ca_a, stride_ca_d,
    # Strides for compact_pruned [B, T, P, D]
    stride_cp_b, stride_cp_t, stride_cp_p, stride_cp_d,
    # Strides for full [B, S, H, D]
    stride_f_b, stride_f_s, stride_f_h, stride_f_d,
    BLOCK_D: tl.constexpr,
):
    # Program ID maps to (batch, seq_pos, head)
    pid = tl.program_id(0)
    # Decode (b, s, h) from flat pid
    h = pid % H
    s = (pid // H) % S
    b = pid // (H * S)

    # Load head metadata
    is_pruned = tl.load(head_source_ptr + h)
    compact_idx = tl.load(head_compact_idx_ptr + h)

    # Output offset base
    out_offset = b * stride_f_b + s * stride_f_s + h * stride_f_h
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    if is_pruned == 0:
        # Active head: read from compact_active[b, s, compact_idx, :]
        in_offset = b * stride_ca_b + s * stride_ca_s + compact_idx * stride_ca_a
        vals = tl.load(compact_active_ptr + in_offset + d_range * stride_ca_d, mask=d_mask)
        tl.store(full_ptr + out_offset + d_range * stride_f_d, vals, mask=d_mask)
    else:
        # Pruned head: check if this is a text position
        text_idx = tl.load(text_seq_map_ptr + s)
        if text_idx >= 0:
            # Text position: read from compact_pruned[b, text_idx, compact_idx, :]
            in_offset = b * stride_cp_b + text_idx * stride_cp_t + compact_idx * stride_cp_p
            vals = tl.load(compact_pruned_ptr + in_offset + d_range * stride_cp_d, mask=d_mask)
            tl.store(full_ptr + out_offset + d_range * stride_f_d, vals, mask=d_mask)
        else:
            # Video position + pruned head: write zeros
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
    Scatter compact active/pruned buffers into full [B, S, H, D] layout.
    """
    B, S, H, D = full.shape
    A = compact_active.shape[2]
    P = compact_pruned.shape[2] if compact_pruned.numel() > 0 else 0
    T = compact_pruned.shape[1] if compact_pruned.numel() > 0 else 0

    # Round D up to next power of 2 for BLOCK_D
    BLOCK_D = triton.next_power_of_2(D)

    grid = (B * S * H,)

    _sparse_scatter_kernel[grid](
        compact_active, compact_pruned, full,
        head_source, head_compact_idx, text_seq_map,
        B, S, H, D, A, P, T,
        # compact_active strides
        compact_active.stride(0), compact_active.stride(1),
        compact_active.stride(2), compact_active.stride(3),
        # compact_pruned strides
        compact_pruned.stride(0), compact_pruned.stride(1),
        compact_pruned.stride(2), compact_pruned.stride(3),
        # full strides
        full.stride(0), full.stride(1),
        full.stride(2), full.stride(3),
        BLOCK_D=BLOCK_D,
    )

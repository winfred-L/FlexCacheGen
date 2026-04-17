import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo

# Sentinel value for pruned video KV head keys. A large finite negative so that
# QK^T produces very negative scores and softmax maps them to ~0.
# We avoid -inf because 0 * -inf = NaN in IEEE 754.
PRUNED_HEAD_KEY_SENTINEL = -1e4


class CacheLayer:
    """
    Per-layer KV cache storage with configurable placement (CPU or GPU).

    Design purpose:
      Each transformer layer owns one CacheLayer instance that persistently holds
      the key and value states. The storage device depends on `offload_kv_to_cpu`:

      - offload=True:  Tensors live on CPU pinned memory. Pinned memory enables
        fast asynchronous DMA transfers to/from GPU via CUDA streams, which is
        the key enabler for CPU offloading without catastrophic bandwidth overhead.

      - offload=False: Tensors live on GPU. This avoids PCIe transfers but uses
        more GPU memory (O(num_layers) instead of O(1)).

      In BOTH modes, the compute-facing format for attention is provided by a
      separate shared GPU buffer managed by KVCacheManager. CacheLayer's storage
      format is deliberately decoupled from the compute format — this allows future
      changes to CacheLayer's internal layout (e.g. sparse storage, head pruning,
      block reordering) without affecting the attention computation path.

    Lifecycle:
      1. Created empty at engine init (one per transformer layer).
      2. `initialize_from_prefill()` — called once after prefill attention for this
         layer. Allocates tensors with capacity (prefill_len + max_new_tokens) and
         copies the prefill KV. Storage device is determined by `offload` param.
      3. `update_from_gpu_buffer()` — called after each decode step. Copies the
         newly-generated token's KV from the shared GPU buffer back to this layer's
         storage, then advances seq_len.
      4. Reset on `KVCacheManager.clear()` between generations.

    Tensor layout: [batch_size=1, max_seq_len, num_kv_heads, head_dim], dtype=bfloat16.
    """

    def __init__(self, max_new_tokens: int):
        self.seq_len: int = 0
        self.max_seq_len: int = 0
        self.max_new_tokens = max_new_tokens
        # Allocated lazily in initialize_from_prefill()
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def initialize_from_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        offload: bool = True,
    ):
        """
        Allocate KV tensors and copy prefill data.

        Called once per layer during prefill, right after attention computes the
        full K/V for all prompt tokens.

        Args:
            key_states:   GPU tensor [1, prefill_len, num_kv_heads, head_dim]
            value_states: GPU tensor, same shape as key_states
            offload:      If True, allocate on CPU pinned memory (for DMA offloading).
                          If False, allocate on the same GPU device as key_states
                          (KV stays resident on GPU).
        """
        batch_size, seq_len, num_heads, head_dim = key_states.shape
        self.seq_len = seq_len
        self.max_seq_len = seq_len + self.max_new_tokens

        if offload:
            # CPU pinned memory: enables fast async DMA to GPU during decode.
            self.keys = torch.empty(
                (batch_size, self.max_seq_len, num_heads, head_dim),
                dtype=key_states.dtype,
                device='cpu',
                pin_memory=True,
            )
            self.values = torch.empty(
                (batch_size, self.max_seq_len, num_heads, head_dim),
                dtype=value_states.dtype,
                device='cpu',
                pin_memory=True,
            )
        else:
            # GPU allocation: KV stays on GPU. Still copied through the shared GPU
            # buffer during decode (not read directly by attention), because future
            # CacheLayer storage formats (sparse, reordered) may differ from the
            # dense layout that flash_attn_with_kvcache expects.
            self.keys = torch.empty(
                (batch_size, self.max_seq_len, num_heads, head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            self.values = torch.empty(
                (batch_size, self.max_seq_len, num_heads, head_dim),
                dtype=value_states.dtype,
                device=value_states.device,
            )

        # Copy prefill KV into the allocated storage.
        self.keys[:, :seq_len].copy_(key_states)
        self.values[:, :seq_len].copy_(value_states)

    def update_from_gpu_buffer(
        self,
        gpu_keys: torch.Tensor,
        gpu_values: torch.Tensor,
    ):
        """
        After decode attention, copy the newly appended token's KV from the shared
        GPU buffer back to this layer's storage, then advance the sequence counter.

        Called in both offload and GPU-resident modes. The shared GPU buffer is the
        single source of truth for newly generated tokens (flash_attn_with_kvcache
        writes there), so we always need to sync back to CacheLayer's own storage.

        Args:
            gpu_keys:   shared GPU buffer keys   [1, max_seq_len, num_kv_heads, head_dim]
            gpu_values: shared GPU buffer values  [1, max_seq_len, num_kv_heads, head_dim]
        """
        pos = self.seq_len
        self.keys[:, pos:pos + 1].copy_(gpu_keys[:, pos:pos + 1])
        self.values[:, pos:pos + 1].copy_(gpu_values[:, pos:pos + 1])
        self.seq_len += 1


class KVCacheManager:
    """
    Manages KV cache across all transformer layers with optional CPU offloading.

    Design purpose:
      Provides a unified API for attention layers regardless of whether KV caches
      are offloaded to CPU or kept on GPU. The behavior is controlled by
      `config.offload_kv_to_cpu`:

      - offload=True:  CacheLayer stores KV on CPU pinned memory. GPU KV memory
        is O(1) — only the shared buffer (1 layer's worth).
      - offload=False: CacheLayer stores KV on GPU. GPU KV memory is
        O(num_layers), but avoids PCIe bandwidth overhead.

      In BOTH modes, decode attention always reads/writes through a shared GPU
      buffer rather than directly from CacheLayer. This decouples the storage
      format from the compute format: CacheLayer can adopt sparse storage, head
      pruning, or block reordering in the future without changing the attention
      code. The shared GPU buffer always holds the dense [B, S, H, D] layout
      that flash_attn_with_kvcache expects.

    Architecture:
      - layers: List[CacheLayer]
          Per-layer KV stores. Device depends on offload mode (CPU pinned vs GPU).
          One per transformer layer (e.g. 32 for Qwen3-VL-8B).
      - _gpu_keys / _gpu_values: torch.Tensor
          A single shared GPU buffer reused across all layers during decode.
          Always allocated (in both modes). Lazily sized on first prefill because
          shape depends on actual prefill length.

    Data flow (offload=True):
      Prefill:  GPU k,v → CacheLayer (CPU pinned). Buffer allocated lazily.
      Decode:   CacheLayer (CPU) → load_layer_to_gpu() → shared GPU buffer
                → flash_attn_with_kvcache → offload_after_decode() → CacheLayer (CPU)

    Data flow (offload=False):
      Prefill:  GPU k,v → CacheLayer (GPU). Buffer allocated lazily.
      Decode:   CacheLayer (GPU) → load_layer_to_gpu() → shared GPU buffer
                → flash_attn_with_kvcache → offload_after_decode() → CacheLayer (GPU)
    """

    def __init__(self, config: Config):
        self.config = config
        self.offload = config.offload_kv_to_cpu
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        # Per-layer KV stores. Initialized empty; populated during prefill.
        self.layers: list[CacheLayer] = [
            CacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)
        ]

        # Shared GPU buffer — one layer's worth of KV, used as the compute-facing
        # staging area for flash_attn_with_kvcache in both modes. Lazily allocated
        # because its shape depends on the actual prefill length.
        self._gpu_keys: torch.Tensor | None = None
        self._gpu_values: torch.Tensor | None = None

    def _ensure_gpu_buffer(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        """
        Lazily allocate the shared GPU buffer on first prefill call.
        Subsequent calls are no-ops. Buffer is sized for one layer's worth of KV.
        """
        if self._gpu_keys is not None:
            return
        self._gpu_keys = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=self.config.device,
            dtype=dtype,
        )
        self._gpu_values = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=self.config.device,
            dtype=dtype,
        )

    def prefill_store_and_offload(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """
        Called during prefill, after attention for layer `layer_idx` has been computed.

        Stores the layer's full KV into the CacheLayer (CPU pinned if offloading,
        GPU if not) and lazily allocates the shared GPU buffer.

        Note: prefill attention itself (flash_attn_func) operates on the raw GPU
        k,v tensors from the Q/K/V projection — it does NOT read from the CacheLayer
        or the shared GPU buffer.
        """
        cache_layer = self.layers[layer_idx]
        cache_layer.initialize_from_prefill(key_states, value_states, offload=self.offload)

        # Lazy-allocate the shared GPU buffer on first layer's prefill call.
        batch_size, _, num_heads, head_dim = key_states.shape
        self._ensure_gpu_buffer(
            batch_size,
            cache_layer.max_seq_len,
            num_heads,
            head_dim,
            key_states.dtype,
        )

    def load_layer_to_gpu(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Copy this layer's KV from CacheLayer storage into the shared GPU buffer
        so that decode attention can read it.

        Always performs a copy — even in GPU-resident mode. This decouples
        CacheLayer's storage format from the dense layout expected by
        flash_attn_with_kvcache, allowing future CacheLayer format changes
        (sparse storage, head pruning, block reordering) without affecting
        the attention computation path.

        In offload mode the copy is CPU→GPU (DMA); in GPU-resident mode the
        copy is GPU→GPU (device-local, fast).

        Returns:
            (gpu_keys, gpu_values, seq_len):
              - gpu_keys, gpu_values: the shared GPU buffer with valid KV data
                in positions [:seq_len]. Passed to flash_attn_with_kvcache.
              - seq_len: current valid sequence length. flash_attn will read
                [:seq_len] and write the new token at position seq_len.
        """
        cache_layer = self.layers[layer_idx]
        seq_len = cache_layer.seq_len

        # Copy valid prefix from CacheLayer to the shared GPU buffer.
        # non_blocking=True is effective for CPU pinned → GPU (async DMA) and
        # is also safe for GPU → GPU (already async on the same stream).
        self._gpu_keys[:, :seq_len].copy_(cache_layer.keys[:, :seq_len], non_blocking=True)
        self._gpu_values[:, :seq_len].copy_(cache_layer.values[:, :seq_len], non_blocking=True)

        # Ensure copy completes before flash_attn reads the buffer.
        # For GPU→GPU on the same stream this is a no-op in practice, but
        # keeps correctness guarantees uniform across both modes.
        torch.cuda.current_stream().synchronize()

        return self._gpu_keys, self._gpu_values, seq_len

    def offload_after_decode(self, layer_idx: int):
        """
        Called after decode attention for layer `layer_idx` completes.

        Copies the newly written token's KV from the shared GPU buffer back to
        this layer's CacheLayer storage and increments seq_len. This runs in
        both offload and GPU-resident modes to keep CacheLayer in sync with
        the buffer (which is the only place flash_attn_with_kvcache writes to).
        """
        self.layers[layer_idx].update_from_gpu_buffer(self._gpu_keys, self._gpu_values)

    def clear(self):
        """
        Reset all per-layer caches and release the shared GPU buffer.
        Called at the end of each generation to free memory before the next request.
        """
        self.layers = [
            CacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)
        ]
        self._gpu_keys = None
        self._gpu_values = None
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, int]:
        """Return KV cache memory usage in bytes."""
        kv_gpu = 0
        kv_cpu = 0
        for layer in self.layers:
            if layer.keys is None:
                continue
            layer_bytes = layer.keys.nbytes + layer.values.nbytes
            if layer.keys.is_cuda:
                kv_gpu += layer_bytes
            else:
                kv_cpu += layer_bytes

        gpu_buffer = 0
        if self._gpu_keys is not None:
            gpu_buffer = self._gpu_keys.nbytes + self._gpu_values.nbytes

        return {
            "kv_cache_gpu_bytes": kv_gpu,
            "kv_cache_cpu_bytes": kv_cpu,
            "gpu_buffer_bytes": gpu_buffer,
        }


class SparseCacheLayer(CacheLayer):
    """
    Per-layer KV cache that stores video and text KV separately.

    Video KV is fixed after prefill; only text KV grows during decode.
    The GPU buffer layout during decode is [video_kv | text_kv], which is
    mathematically equivalent to the original ordering.
    """

    def __init__(self, max_new_tokens: int, paged: bool = False, page_size: int = 256):
        # Skip CacheLayer.__init__ to avoid conflict with seq_len property.
        self.max_seq_len: int = 0
        self.max_new_tokens = max_new_tokens
        self.video_keys: torch.Tensor | None = None
        self.video_values: torch.Tensor | None = None
        self.text_keys: torch.Tensor | None = None
        self.text_values: torch.Tensor | None = None
        self.video_len: int = 0
        self.text_seq_len: int = 0

        # --- Static head pruning metadata ---
        # When static head pruning is active, only a subset of KV heads are stored
        # for video tokens (text KV always uses all heads). These fields are set by
        # SparseKVCacheManager via set_pruning_info() before prefill begins.
        self.kept_head_indices: torch.Tensor | None = None  # 1-D long tensor of head indices to keep
        self.num_total_heads: int = 0                       # total KV heads before pruning (e.g. 8)
        self.is_pruned: bool = False                        # convenience flag: True if any heads pruned

        # --- Paged attention metadata (for Quest seq-dimension sparsity) ---
        # Active when config.dynamic_sparse_threshold is not None.
        # Video KV is stored in paged format; text KV remains continuous.
        self.paged: bool = paged
        self.page_size: int = page_size
        self.num_video_pages: int = 0
        self.video_paged_keys: torch.Tensor | None = None   # [num_pages, page_size, num_heads, D]
        self.video_paged_values: torch.Tensor | None = None
        self.page_min_keys: torch.Tensor | None = None      # [num_pages, num_heads, D] - Quest metadata
        self.page_max_keys: torch.Tensor | None = None      # [num_pages, num_heads, D] - Quest metadata

    def set_pruning_info(self, pruned_heads: list[int], num_total_heads: int):
        """
        Configure per-layer head pruning for video KV.

        Called by SparseKVCacheManager during initialization (and after clear()).
        Computes which heads to keep as the complement of pruned_heads within
        [0, num_total_heads). If pruned_heads is empty, pruning is disabled for
        this layer and all heads are stored.

        Args:
            pruned_heads:    List of head indices to prune (e.g. [2, 4, 5, 6]).
            num_total_heads: Total number of KV heads in the model (e.g. 8).
        """
        self.num_total_heads = num_total_heads
        if not pruned_heads:
            # No pruning for this layer — store all heads as usual.
            self.is_pruned = False
            self.kept_head_indices = None
            return
        self.is_pruned = True
        # Compute kept heads as the sorted complement of pruned heads.
        kept = sorted(set(range(num_total_heads)) - set(pruned_heads))
        self.kept_head_indices = torch.tensor(kept, dtype=torch.long)

    @property
    def seq_len(self) -> int:
        return self.video_len + self.text_seq_len

    def _compute_page_metadata(self, video_keys_flat: torch.Tensor, video_len: int):
        """
        Compute per-page min/max key metadata for Quest criticality scoring.

        Stored on the same device as video_keys_flat (GPU) so scoring can run
        in a single GPU kernel per layer without CPU↔GPU ping-pong. Metadata
        is small (num_pages × num_heads × head_dim × 2) — a few MB per layer.

        Args:
            video_keys_flat: [video_len, num_heads, head_dim] - video key tensor (on GPU)
            video_len:       actual number of valid video tokens (excludes padding)
        """
        num_h = video_keys_flat.shape[1]
        head_dim = video_keys_flat.shape[2]
        device = video_keys_flat.device
        dtype = video_keys_flat.dtype

        page_min = torch.empty(self.num_video_pages, num_h, head_dim, dtype=dtype, device=device)
        page_max = torch.empty(self.num_video_pages, num_h, head_dim, dtype=dtype, device=device)

        full_pages = video_len // self.page_size
        remainder = video_len % self.page_size

        if full_pages > 0:
            full_data = video_keys_flat[:full_pages * self.page_size].view(
                full_pages, self.page_size, num_h, head_dim
            )
            page_min[:full_pages] = full_data.amin(dim=1)
            page_max[:full_pages] = full_data.amax(dim=1)

        if remainder > 0:
            last_data = video_keys_flat[full_pages * self.page_size:video_len]
            page_min[full_pages] = last_data.amin(dim=0)
            page_max[full_pages] = last_data.amax(dim=0)

        self.page_min_keys = page_min
        self.page_max_keys = page_max

    def quest_score_pages(self, q_kv: torch.Tensor) -> torch.Tensor:
        """
        Quest criticality score per video page — upper bound of sum_i q[i]·k[i].

        For each dim: U_i = max(q[i]·m[i], q[i]·M[i]) ≥ q[i]·k[i] for any k in
        the page (where m/M are this page's min/max). Summing U_i over the head
        dim gives a per-head upper bound on Q·K for tokens in the page; summing
        across heads gives a page-level score. Higher score → page more likely
        to contain high-attention tokens.

        Args:
            q_kv: [num_heads, head_dim] — query already mapped to KV groups and
                  restricted to this layer's kept heads, matching the head
                  layout of page_min_keys / page_max_keys. Must live on the
                  same device as the metadata.
        Returns:
            scores: [num_video_pages]
        """
        q = q_kv.unsqueeze(0)  # [1, H, D]
        prod_min = q * self.page_min_keys  # [P, H, D]
        prod_max = q * self.page_max_keys  # [P, H, D]
        upper = torch.maximum(prod_min, prod_max)  # [P, H, D]
        # sum over head_dim (per-head Q·K upper bound), then aggregate over heads
        return upper.sum(dim=-1).sum(dim=-1)  # [P]

    def initialize_from_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        offload: bool = True,
        video_indices: torch.Tensor | None = None,
        text_indices: torch.Tensor | None = None,
    ):
        """
        Split prefill KV into video and text parts and store separately.

        Args:
            key_states:    GPU tensor [1, prefill_len, num_kv_heads, head_dim]
            value_states:  GPU tensor, same shape
            offload:       CPU pinned memory if True, GPU if False
            video_indices: 1-D tensor of video token positions in the prefill sequence
            text_indices:  1-D tensor of text token positions in the prefill sequence
        """
        batch_size, prefill_len, num_heads, head_dim = key_states.shape

        video_len = video_indices.shape[0]
        text_len = text_indices.shape[0]
        assert video_len + text_len == prefill_len

        self.video_len = video_len
        self.text_seq_len = text_len
        self.max_seq_len = prefill_len + self.max_new_tokens

        # Extract video and text KV from the interleaved prefill tensor
        video_k = key_states[:, video_indices]    # [1, video_len, H, D]
        video_v = value_states[:, video_indices]
        text_k = key_states[:, text_indices]       # [1, text_len, H, D]
        text_v = value_states[:, text_indices]

        alloc_kwargs = dict(dtype=key_states.dtype)
        if offload:
            alloc_kwargs.update(device='cpu', pin_memory=True)
        else:
            alloc_kwargs.update(device=key_states.device)

        # --- Allocate video KV storage ---
        # When head pruning is active, only the kept heads are stored, producing a
        # compressed tensor with shape [B, video_len, num_kept_heads, D] instead of
        # the full [B, video_len, num_total_heads, D]. This reduces CPU/GPU memory
        # proportional to the per-layer pruning ratio. Text KV is never pruned.
        if self.is_pruned:
            # Select only the kept heads from the extracted video KV.
            # kept_head_indices is a CPU long tensor (e.g. [0, 1, 3, 7]).
            # PyTorch allows a CPU index tensor to fancy-index a GPU tensor.
            num_kept = self.kept_head_indices.shape[0]
            video_k_kept = video_k[:, :, self.kept_head_indices, :]  # [1, video_len, num_kept, D]
            video_v_kept = video_v[:, :, self.kept_head_indices, :]
            num_h = num_kept
        else:
            video_k_kept = video_k
            video_v_kept = video_v
            num_h = num_heads

        if self.paged:
            # --- Paged video KV storage ---
            # Reshape video KV into pages of size page_size. The last page is
            # zero-padded to the page boundary (padding tokens have key=0 →
            # QK^T≈0 → negligible attention weight, consistent with sentinel).
            self.num_video_pages = (video_len + self.page_size - 1) // self.page_size
            padded_len = self.num_video_pages * self.page_size

            # Remove batch dim: [1, video_len, H, D] → [video_len, H, D]
            vk = video_k_kept.squeeze(0)
            vv = video_v_kept.squeeze(0)

            # Compute page metadata BEFORE padding (only valid tokens)
            self._compute_page_metadata(vk, video_len)

            # Pad to page boundary and reshape to [num_pages, page_size, H, D]
            if padded_len > video_len:
                pad_k = torch.zeros(padded_len - video_len, num_h, head_dim,
                                    dtype=vk.dtype, device=vk.device)
                pad_v = torch.zeros(padded_len - video_len, num_h, head_dim,
                                    dtype=vv.dtype, device=vv.device)
                vk = torch.cat([vk, pad_k], dim=0)
                vv = torch.cat([vv, pad_v], dim=0)

            vk = vk.view(self.num_video_pages, self.page_size, num_h, head_dim)
            vv = vv.view(self.num_video_pages, self.page_size, num_h, head_dim)

            # Allocate and copy to storage (CPU pinned or GPU)
            self.video_paged_keys = torch.empty_like(vk, **alloc_kwargs)
            self.video_paged_values = torch.empty_like(vv, **alloc_kwargs)
            self.video_paged_keys.copy_(vk)
            self.video_paged_values.copy_(vv)
        else:
            # --- Continuous video KV storage (original path) ---
            self.video_keys = torch.empty(
                (batch_size, video_len, num_h, head_dim), **alloc_kwargs,
            )
            self.video_values = torch.empty(
                (batch_size, video_len, num_h, head_dim), **alloc_kwargs,
            )
            self.video_keys.copy_(video_k_kept)
            self.video_values.copy_(video_v_kept)

        # Text storage: prefill text + room for max_new_tokens
        text_max_len = text_len + self.max_new_tokens
        self.text_keys = torch.empty(
            (batch_size, text_max_len, num_heads, head_dim), **alloc_kwargs,
        )
        self.text_values = torch.empty(
            (batch_size, text_max_len, num_heads, head_dim), **alloc_kwargs,
        )
        self.text_keys[:, :text_len].copy_(text_k)
        self.text_values[:, :text_len].copy_(text_v)

    def update_from_gpu_buffer(
        self,
        gpu_keys: torch.Tensor,
        gpu_values: torch.Tensor,
    ):
        """
        Copy new text token's KV from GPU buffer back to text storage.

        In the GPU buffer, new token sits at position video_len + text_seq_len.
        """
        buf_pos = self.video_len + self.text_seq_len
        self.text_keys[:, self.text_seq_len:self.text_seq_len + 1].copy_(
            gpu_keys[:, buf_pos:buf_pos + 1]
        )
        self.text_values[:, self.text_seq_len:self.text_seq_len + 1].copy_(
            gpu_values[:, buf_pos:buf_pos + 1]
        )
        self.text_seq_len += 1


class SparseKVCacheManager(KVCacheManager):
    """
    KV cache manager that stores video and text KV separately per layer.

    GPU buffer layout during decode: [video_kv | text_kv].
    Video KV is fixed after prefill; only text KV region grows.

    When config.dynamic_sparse_threshold is not None, paged attention is enabled:
    - Video KV is stored in paged format [num_pages, page_size, H, D] per layer
    - GPU buffer is a paged pool [total_blocks, page_size, H, D]
    - block_table maps logical pages to physical blocks for flash_attn_with_kvcache
    - Page metadata (min/max keys) is maintained for future Quest scoring
    """

    def __init__(self, config: Config):
        # Skip KVCacheManager.__init__ to use SparseCacheLayer instead
        self.config = config
        self.offload = config.offload_kv_to_cpu
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        # --- Paged attention configuration ---
        self.paged: bool = config.dynamic_sparse_threshold is not None
        self.page_size: int = config.page_size

        self.layers: list[SparseCacheLayer] = [
            SparseCacheLayer(self.config.max_new_tokens, paged=self.paged, page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]

        # Continuous GPU buffer (non-paged mode)
        self._gpu_keys: torch.Tensor | None = None
        self._gpu_values: torch.Tensor | None = None

        # Paged GPU buffer and block table (paged mode)
        self._gpu_paged_keys: torch.Tensor | None = None
        self._gpu_paged_values: torch.Tensor | None = None
        self._block_table: torch.Tensor | None = None  # [1, total_blocks], int32
        self.block_table: torch.Tensor | None = None    # set per load_layer_to_gpu(), read by attention

        # Paged layout metadata (set during _ensure_gpu_buffer in paged mode)
        self.num_video_pages: int = 0
        self.video_padded_len: int = 0
        self.total_blocks: int = 0

        # Video metadata — set by set_video_info() before prefill
        self.video_info: VideoInfo | None = None
        self.video_indices: torch.Tensor | None = None
        self.text_indices: torch.Tensor | None = None

        # --- Static head pruning configuration ---
        # If config provides a prune-heads map ({layer_idx: [head_indices]}), apply
        # it to each SparseCacheLayer so that video KV storage is compressed.
        self.prune_heads_map: dict[int, list[int]] | None = config.static_sparse_prune_heads
        self._apply_pruning_info()

        # --- Quest (dynamic seq-dimension sparsity) ---
        # Active iff dynamic_sparse_threshold is not None (== self.paged). The
        # threshold is interpreted as the fraction of video pages to KEEP
        # (higher = denser, 1.0 = load all, matching the non-Quest paged path).
        # Set per-decode-step by Qwen3VLTextAttention via set_current_query().
        self.num_q_heads: int = self.config.hf_config.text_config.num_attention_heads
        self.num_kv_heads: int = self.config.hf_config.text_config.num_key_value_heads
        self.current_query: torch.Tensor | None = None

    def set_current_query(self, q: torch.Tensor):
        """
        Register the current decode-step query so the next load_layer_to_gpu()
        can run Quest page selection. Called by Qwen3VLTextAttention before
        each decode-step load. A no-op value (None) falls back to loading all
        video pages in paged mode.

        Args:
            q: [1, 1, num_attention_heads, head_dim] — post-RoPE query for the
               new token being decoded (on GPU).
        """
        self.current_query = q

    def _map_q_to_kv_heads(self, q: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multi-head Q into per-KV-head Q for Quest scoring.

        GQA collapses a group of Q heads onto one shared KV head. We average
        within each group so a single [num_kv_heads, head_dim] query represents
        the group and can be matched against the per-KV-head page metadata.

        Args:
            q: [1, 1, num_q_heads, head_dim]
        Returns:
            [num_kv_heads, head_dim]
        """
        head_dim = q.shape[-1]
        if self.num_q_heads == self.num_kv_heads:
            return q.reshape(self.num_kv_heads, head_dim)
        if self.num_q_heads % self.num_kv_heads == 0:
            gqa_ratio = self.num_q_heads // self.num_kv_heads
            return (
                q.view(1, 1, self.num_kv_heads, gqa_ratio, head_dim)
                 .mean(dim=3)
                 .reshape(self.num_kv_heads, head_dim)
            )
        # Non-divisible ratio fallback: broadcast global Q mean to each KV head.
        q_mean = q.mean(dim=2)  # [1, 1, head_dim]
        return q_mean.expand(1, 1, self.num_kv_heads, head_dim).reshape(self.num_kv_heads, head_dim)

    def _apply_pruning_info(self):
        """
        Propagate static head pruning configuration from config to each SparseCacheLayer.

        For each layer, looks up the list of heads to prune from self.prune_heads_map.
        Layers not present in the map (or with an empty list) will store all heads.
        This is called during __init__ and after clear() to reconfigure fresh layers.
        """
        if self.prune_heads_map is None:
            return
        num_kv_heads = self.config.hf_config.text_config.num_key_value_heads
        for layer_idx, layer in enumerate(self.layers):
            pruned = self.prune_heads_map.get(layer_idx, [])
            layer.set_pruning_info(pruned, num_kv_heads)

    def _ensure_gpu_buffer(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        if not self.paged:
            # Non-paged mode: use parent's continuous buffer allocation.
            super()._ensure_gpu_buffer(batch_size, max_seq_len, num_heads, head_dim, dtype)
            return

        # --- Paged mode: allocate paged pool + block table ---
        if self._gpu_paged_keys is not None:
            return  # already allocated

        video_len = len(self.video_indices)
        text_len = len(self.text_indices)

        self.num_video_pages = (video_len + self.page_size - 1) // self.page_size
        self.video_padded_len = self.num_video_pages * self.page_size
        num_text_pages = (text_len + self.config.max_new_tokens + self.page_size - 1) // self.page_size
        self.total_blocks = self.num_video_pages + num_text_pages

        self._gpu_paged_keys = torch.zeros(
            (self.total_blocks, self.page_size, num_heads, head_dim),
            device=self.config.device, dtype=dtype,
        )
        self._gpu_paged_values = torch.zeros(
            (self.total_blocks, self.page_size, num_heads, head_dim),
            device=self.config.device, dtype=dtype,
        )

        # Identity block table: logical page i → physical block i
        self._block_table = torch.arange(
            self.total_blocks, dtype=torch.int32, device=self.config.device,
        ).unsqueeze(0)  # [1, total_blocks]

    def set_video_info(self, video_info: VideoInfo, prefill_len: int):
        """
        Precompute video and text index tensors from VideoInfo.

        Args:
            video_info: VideoInfo from get_video_info()
            prefill_len: total number of tokens in the prefill sequence
        """
        self.video_info = video_info

        # Build set of all video token positions from index_ranges
        video_pos = set()
        for start_idx, end_idx in video_info.index_ranges:
            video_pos.update(range(start_idx, end_idx + 1))  # end_idx is inclusive

        video_list = sorted(video_pos)
        text_list = [i for i in range(prefill_len) if i not in video_pos]

        self.video_indices = torch.tensor(video_list, dtype=torch.long)
        self.text_indices = torch.tensor(text_list, dtype=torch.long)

    def prefill_store_and_offload(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        cache_layer = self.layers[layer_idx]
        cache_layer.initialize_from_prefill(
            key_states, value_states,
            offload=self.offload,
            video_indices=self.video_indices,
            text_indices=self.text_indices,
        )

        # Lazy-allocate shared GPU buffer (same as parent)
        batch_size, _, num_heads, head_dim = key_states.shape
        self._ensure_gpu_buffer(
            batch_size,
            cache_layer.max_seq_len,
            num_heads,
            head_dim,
            key_states.dtype,
        )

    def load_layer_to_gpu(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Reconstruct GPU buffer for decode attention.

        Non-paged mode: dense continuous buffer [B, seq_len, H, D].
        Paged mode: paged pool [total_blocks, page_size, H, D] with block_table.
        """
        cache_layer = self.layers[layer_idx]
        video_len = cache_layer.video_len
        text_seq_len = cache_layer.text_seq_len

        if not self.paged:
            return self._load_layer_continuous(cache_layer, video_len, text_seq_len)
        else:
            return self._load_layer_paged(cache_layer, video_len, text_seq_len)

    def _load_layer_continuous(
        self,
        cache_layer: SparseCacheLayer,
        video_len: int,
        text_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Non-paged path: reconstruct dense GPU buffer as [video_kv | text_kv]."""
        total_len = video_len + text_seq_len
        self.block_table = None

        # --- Video KV region [0:video_len] ---
        if cache_layer.is_pruned:
            self._gpu_keys[:, :video_len].fill_(PRUNED_HEAD_KEY_SENTINEL)
            self._gpu_values[:, :video_len].fill_(0.0)

            kept = cache_layer.kept_head_indices.tolist()
            for i, h in enumerate(kept):
                self._gpu_keys[:, :video_len, h:h+1, :].copy_(
                    cache_layer.video_keys[:, :, i:i+1, :], non_blocking=True
                )
                self._gpu_values[:, :video_len, h:h+1, :].copy_(
                    cache_layer.video_values[:, :, i:i+1, :], non_blocking=True
                )
        else:
            self._gpu_keys[:, :video_len].copy_(cache_layer.video_keys, non_blocking=True)
            self._gpu_values[:, :video_len].copy_(cache_layer.video_values, non_blocking=True)

        # --- Text KV region [video_len:total_len] --- (never pruned)
        self._gpu_keys[:, video_len:total_len].copy_(
            cache_layer.text_keys[:, :text_seq_len], non_blocking=True
        )
        self._gpu_values[:, video_len:total_len].copy_(
            cache_layer.text_values[:, :text_seq_len], non_blocking=True
        )

        torch.cuda.current_stream().synchronize()
        return self._gpu_keys, self._gpu_values, total_len

    def _load_layer_paged(
        self,
        cache_layer: SparseCacheLayer,
        video_len: int,
        text_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Paged path with Quest query-aware page selection for the video region.

        Given the current decode query (set via set_current_query()), each call
        scores the layer's video pages using the stored min/max key metadata,
        keeps the Top-K pages (K = ceil(num_video_pages * threshold)), and loads
        only those into the compact physical block prefix [0..K-1]. Text KV is
        never sparsified — all text pages are loaded into their fixed physical
        slots [num_video_pages..]. The block_table rewires logical pages so the
        kernel sees Top-K video pages followed by text pages.

        Fallback: if current_query is unset (non-decode call, or attention did
        not populate it), load all video pages — behavior reduces to the
        original paged path.

        cache_seqlens = K * page_size + text_seq_len (K < num_video_pages under
        Quest, == num_video_pages in fallback). Padding in the last video page
        contributes keys=0 which yield softmax≈0 weight — same harmless effect
        as in the non-Quest path.
        """
        num_video_pages = cache_layer.num_video_pages
        page_size = self.page_size
        threshold = self.config.dynamic_sparse_threshold

        # === Quest: select Top-K video pages for this layer's current Q ===
        use_quest = (
            num_video_pages > 0
            and threshold is not None
            and self.current_query is not None
            and cache_layer.page_min_keys is not None
        )
        if use_quest:
            top_k = max(1, min(num_video_pages, int(round(num_video_pages * threshold))))
            if top_k < num_video_pages:
                # Map Q heads → KV heads; restrict to kept heads to match metadata layout
                q_kv_full = self._map_q_to_kv_heads(self.current_query)  # [num_kv_heads, D]
                if cache_layer.is_pruned:
                    kept_idx = cache_layer.kept_head_indices.to(q_kv_full.device)
                    q_kv = q_kv_full.index_select(0, kept_idx)
                else:
                    q_kv = q_kv_full
                q_kv = q_kv.to(dtype=cache_layer.page_min_keys.dtype)

                scores = cache_layer.quest_score_pages(q_kv)                 # [num_video_pages]
                selected_pages = torch.topk(scores, top_k).indices            # [top_k]
                # Sort so physical block order follows original page order (slightly
                # better locality for the kernel; also makes debugging easier).
                selected_pages, _ = torch.sort(selected_pages)
            else:
                # threshold keeps everything — skip scoring
                selected_pages = None
                top_k = num_video_pages
        else:
            selected_pages = None
            top_k = num_video_pages

        num_video_blocks_used = top_k

        # --- Gather source video pages from CacheLayer storage ---
        if selected_pages is None:
            src_k_view = cache_layer.video_paged_keys                        # [num_video_pages, S, H_stored, D]
            src_v_view = cache_layer.video_paged_values
        else:
            sel_dev = selected_pages.to(cache_layer.video_paged_keys.device)
            src_k_view = cache_layer.video_paged_keys.index_select(0, sel_dev)
            src_v_view = cache_layer.video_paged_values.index_select(0, sel_dev)

        # --- Load selected video pages into compact physical prefix [0..K-1] ---
        # Reordering video pages is safe: RoPE is already baked into each K vector,
        # so attention scores don't depend on the kernel's per-page traversal order.
        if cache_layer.is_pruned:
            self._gpu_paged_keys[:num_video_blocks_used].fill_(PRUNED_HEAD_KEY_SENTINEL)
            self._gpu_paged_values[:num_video_blocks_used].fill_(0.0)

            kept = cache_layer.kept_head_indices.tolist()
            for i, h in enumerate(kept):
                self._gpu_paged_keys[:num_video_blocks_used, :, h:h+1, :].copy_(
                    src_k_view[:, :, i:i+1, :], non_blocking=True
                )
                self._gpu_paged_values[:num_video_blocks_used, :, h:h+1, :].copy_(
                    src_v_view[:, :, i:i+1, :], non_blocking=True
                )
        else:
            self._gpu_paged_keys[:num_video_blocks_used].copy_(src_k_view, non_blocking=True)
            self._gpu_paged_values[:num_video_blocks_used].copy_(src_v_view, non_blocking=True)

        # --- Text pages [block num_video_pages..] — always fully loaded ---
        if text_seq_len > 0:
            text_k = cache_layer.text_keys[0, :text_seq_len]
            text_v = cache_layer.text_values[0, :text_seq_len]

            num_text_pages_used = (text_seq_len + page_size - 1) // page_size
            for p in range(num_text_pages_used):
                t_start = p * page_size
                t_end = min((p + 1) * page_size, text_seq_len)
                physical_block = num_video_pages + p

                self._gpu_paged_keys[physical_block, :t_end - t_start].copy_(
                    text_k[t_start:t_end], non_blocking=True
                )
                self._gpu_paged_values[physical_block, :t_end - t_start].copy_(
                    text_v[t_start:t_end], non_blocking=True
                )

        # --- Build block_table ---
        # Logical pages: [0..K-1] = selected video pages (compact, physical 0..K-1),
        #                [K..K+T]  = text pages (physical num_video_pages..).
        # The page holding the new token (logical index K + text_seq_len//page_size)
        # must be present so flash_attn can write into it — add it explicitly.
        seqlens_int = num_video_blocks_used * page_size + text_seq_len
        cache_seqlens = torch.tensor(
            [seqlens_int], dtype=torch.int32, device=self.config.device,
        )

        num_text_pages_ref = text_seq_len // page_size + 1  # +1 for new token's page
        video_block_ids = torch.arange(
            num_video_blocks_used, dtype=torch.int32, device=self.config.device,
        )
        text_block_ids = torch.arange(
            num_video_pages,
            num_video_pages + num_text_pages_ref,
            dtype=torch.int32,
            device=self.config.device,
        )
        self.block_table = torch.cat([video_block_ids, text_block_ids]).unsqueeze(0)  # [1, L]

        torch.cuda.current_stream().synchronize()
        return self._gpu_paged_keys, self._gpu_paged_values, cache_seqlens

    def offload_after_decode(self, layer_idx: int):
        cache_layer = self.layers[layer_idx]

        if not self.paged:
            cache_layer.update_from_gpu_buffer(self._gpu_keys, self._gpu_values)
            return

        # --- Paged mode: read new text token from paged GPU buffer ---
        # flash_attn_with_kvcache wrote the new token at logical position
        # cache_seqlens = video_padded_len + text_seq_len, which maps to:
        #   page  = num_video_pages + text_seq_len // page_size
        #   offset = text_seq_len % page_size
        text_seq_len = cache_layer.text_seq_len
        text_page_idx = text_seq_len // self.page_size
        text_offset = text_seq_len % self.page_size
        physical_block = cache_layer.num_video_pages + text_page_idx

        cache_layer.text_keys[0, text_seq_len:text_seq_len + 1].copy_(
            self._gpu_paged_keys[physical_block, text_offset:text_offset + 1]
        )
        cache_layer.text_values[0, text_seq_len:text_seq_len + 1].copy_(
            self._gpu_paged_values[physical_block, text_offset:text_offset + 1]
        )
        cache_layer.text_seq_len += 1

    def clear(self):
        self.layers = [
            SparseCacheLayer(self.config.max_new_tokens, paged=self.paged, page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]
        # Re-apply head pruning config to freshly created layers.
        self._apply_pruning_info()
        self._gpu_keys = None
        self._gpu_values = None
        self._gpu_paged_keys = None
        self._gpu_paged_values = None
        self._block_table = None
        self.block_table = None
        self.num_video_pages = 0
        self.video_padded_len = 0
        self.total_blocks = 0
        self.video_info = None
        self.video_indices = None
        self.text_indices = None
        self.current_query = None
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, int]:
        """Return KV cache memory usage with video/text breakdown."""
        video_bytes = 0
        text_bytes = 0
        kv_gpu = 0
        kv_cpu = 0
        for layer in self.layers:
            # Video bytes: paged or continuous storage
            if layer.video_paged_keys is not None:
                v_bytes = layer.video_paged_keys.nbytes + layer.video_paged_values.nbytes
            elif layer.video_keys is not None:
                v_bytes = layer.video_keys.nbytes + layer.video_values.nbytes
            else:
                continue
            t_bytes = layer.text_keys.nbytes + layer.text_values.nbytes
            video_bytes += v_bytes
            text_bytes += t_bytes
            layer_bytes = v_bytes + t_bytes

            # Determine device from whichever video storage is active
            video_storage = layer.video_paged_keys if layer.video_paged_keys is not None else layer.video_keys
            if video_storage.is_cuda:
                kv_gpu += layer_bytes
            else:
                kv_cpu += layer_bytes

        gpu_buffer = 0
        if self._gpu_keys is not None:
            gpu_buffer = self._gpu_keys.nbytes + self._gpu_values.nbytes
        if self._gpu_paged_keys is not None:
            gpu_buffer += self._gpu_paged_keys.nbytes + self._gpu_paged_values.nbytes

        return {
            "kv_cache_gpu_bytes": kv_gpu,
            "kv_cache_cpu_bytes": kv_cpu,
            "gpu_buffer_bytes": gpu_buffer,
            "video_kv_bytes": video_bytes,
            "text_kv_bytes": text_bytes,
        }

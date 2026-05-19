import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo

# Legacy sentinel constant for pruned video KV head keys.
# Superseded by config.pruning_k_type which controls sentinel behavior:
#   "zero":     K filled with 0.0 (legacy, softmax ≈ 1/N, not true pruning)
#   "negative": K filled with -M*sign(Q), giving Q·K ≪ 0 → softmax ≈ 0
PRUNED_HEAD_KEY_SENTINEL = 0.0


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

        # --- Quest / dynamic sparsity metadata ---
        # Active when config.dynamic_sparse_threshold is not None.
        # Video KV is stored in paged format [num_pages, page_size, H, D] as source;
        # _load_layer_quest gathers selected pages into a contiguous GPU buffer.
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

    def _compute_page_metadata(self, video_keys_flat: torch.Tensor, video_len: int,
                               store_on_cpu: bool = False):
        """
        Compute per-page min/max key metadata for Quest criticality scoring.

        When store_on_cpu is True (CPU offloading mode), metadata is stored on
        CPU pinned memory so Quest scoring can run entirely on CPU — avoiding the
        GPU→CPU sync that would otherwise be needed to transfer selected page
        indices back for CPU-side index_select. Metadata is small
        (num_pages × num_heads × head_dim × 2) — a few MB per layer.

        Args:
            video_keys_flat: [video_len, num_heads, head_dim] - video key tensor (on GPU)
            video_len:       actual number of valid video tokens (excludes padding)
            store_on_cpu:    If True, store metadata on CPU pinned memory
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

        # Store on CPU pinned memory when offloading — avoids GPU→CPU sync during decode
        if store_on_cpu:
            page_min = page_min.cpu().pin_memory()
            page_max = page_max.cpu().pin_memory()

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
        store_metadata_on_cpu: bool = False,
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
            self._compute_page_metadata(vk, video_len, store_on_cpu=store_metadata_on_cpu)

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

    When config.dynamic_sparse_threshold is not None, Quest dynamic sparsity is enabled:
    - Video KV is loaded selectively: Top-K pages chosen by Quest scoring
    - GPU buffer is a contiguous 3D tensor [B, max_seq_len, H, D]
    """

    def __init__(self, config: Config):
        # Skip KVCacheManager.__init__ to use SparseCacheLayer instead
        self.config = config
        self.offload = config.offload_kv_to_cpu
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        # --- Quest / dynamic sparsity configuration ---
        # When dynamic_sparse_threshold is not None, use Quest page selection
        # with a contiguous 3D buffer (not paged attention).
        self.paged: bool = config.dynamic_sparse_threshold is not None
        self.page_size: int = config.page_size  # used for video page size in Quest

        self.layers: list[SparseCacheLayer] = [
            SparseCacheLayer(self.config.max_new_tokens, paged=self.paged, page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]

        # Continuous GPU buffer (shared by both paged and non-paged modes)
        self._gpu_keys: torch.Tensor | None = None
        self._gpu_values: torch.Tensor | None = None

        # Video metadata — set by set_video_info() before prefill
        self.video_info: VideoInfo | None = None
        self.video_indices: torch.Tensor | None = None
        self.text_indices: torch.Tensor | None = None

        # --- Static head pruning configuration ---
        # If config provides a prune-heads map ({layer_idx: [head_indices]}), apply
        # it to each SparseCacheLayer so that video KV storage is compressed.
        self.prune_heads_map: dict[int, list[int]] | None = config.static_sparse_prune_heads
        self.pruning_k_type: str = getattr(config, 'pruning_k_type', 'zero')
        self._apply_pruning_info()

        # --- Quest (dynamic seq-dimension sparsity) ---
        # Active iff dynamic_sparse_threshold is not None (== self.paged). The
        # threshold is the fraction of video pages to KEEP (higher = denser,
        # 1.0 = load all). Uses a contiguous 3D GPU buffer (not paged attention).
        # Set per-decode-step by Qwen3VLTextAttention via set_current_query().
        self.num_q_heads: int = self.config.hf_config.text_config.num_attention_heads
        self.num_kv_heads: int = self.config.hf_config.text_config.num_key_value_heads
        self.current_query: torch.Tensor | None = None

        # --- Pipeline: overlap attention with KV prefetch ---
        # Active iff config.pipeline is True (requires dynamic_sparse_threshold).
        # Uses dual GPU buffers (ping-pong) and a dedicated CUDA stream for
        # async DMA transfers. After computing Q for layer L, the pipeline
        # starts prefetching layer (L+1)'s KV on the prefetch stream while
        # layer L's attention+MLP runs on the default stream.
        self._pipeline: bool = False
        self._gpu_keys_alt: torch.Tensor | None = None
        self._gpu_values_alt: torch.Tensor | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._q_ready_event: torch.cuda.Event | None = None
        self._current_pipeline_slot: int = 0  # set by decoding_pipeline per layer
        # Per-slot state: tracks buffer layout for offload_after_decode_pipeline
        # slot 0 = primary buffer, slot 1 = alt buffer
        self._slot_info: list[dict] = [
            {"cache_seqlens": 0, "video_tokens_used": 0},
            {"cache_seqlens": 0, "video_tokens_used": 0},
        ]

    def set_current_query(self, q: torch.Tensor):
        """
        Register the current decode-step query so the next load_layer_to_gpu()
        can run Quest page selection. Called by Qwen3VLTextAttention before
        each decode-step load. A no-op value (None) falls back to loading all
        video pages (no Quest sparsity).

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

    def _get_pruned_key_sentinel(self) -> torch.Tensor | float:
        """
        Compute the K sentinel value for pruned attention heads.

        The sentinel is derived from self.current_query, which is set by the
        attention layer via set_current_query(). In sequential decode this is
        always the current layer's own Q. In pipeline mode, the prefetch stream
        fills the sentinel with the previous layer's Q; _refill_pruned_sentinel()
        then corrects it on the default stream once the current layer's Q is
        available.

        Returns:
            If pruning_k_type == "zero" or current_query is None: 0.0 (float).
            If pruning_k_type == "negative": [num_kv_heads, head_dim] tensor where
                each element is -M * sign(Q[head, dim]), so that Q·K is a large
                negative value → softmax ≈ 0, truly eliminating the pruned head.
        """
        if self.pruning_k_type == "zero" or self.current_query is None:
            return 0.0
        # "negative" mode: per-element sentinel driven by query sign
        q_kv = self._map_q_to_kv_heads(self.current_query)  # [num_kv_heads, head_dim]
        return -1e4 * torch.sign(q_kv)  # [num_kv_heads, head_dim]

    def _fill_pruned_sentinel(
        self,
        target_keys: torch.Tensor,
        target_values: torch.Tensor,
        num_tokens: int,
    ):
        """
        Fill the video region of the GPU buffer with pruning sentinel values.

        For pruning_k_type="zero": keys are filled with 0.0, values with 0.0.
        For pruning_k_type="negative": keys are filled with -M*sign(Q) per
        element so that Q·K is a large negative → softmax ≈ 0.

        In pipeline mode this runs on the prefetch stream with the PREVIOUS
        layer's Q (the next layer's Q is not yet available). The mismatch is
        corrected later by _refill_pruned_sentinel() on the default stream
        once the current layer's Q is computed.

        Args:
            target_keys:   GPU key buffer [1, max_seq_len, num_kv_heads, head_dim]
            target_values: GPU value buffer, same shape
            num_tokens:    number of video tokens to fill (video_len or video_tokens_used)
        """
        sentinel = self._get_pruned_key_sentinel()
        if isinstance(sentinel, float):
            target_keys[:, :num_tokens].fill_(sentinel)
        else:
            # sentinel is [num_kv_heads, head_dim]; broadcast to buffer shape
            target_keys[:, :num_tokens] = sentinel[None, None, :, :]
        target_values[:, :num_tokens].fill_(0.0)

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

        # Paged mode (Quest): also uses a single contiguous buffer [B, max_seq_len, H, D].
        # Quest dynamically selects top-K video pages and gathers them into the buffer prefix.
        if self._gpu_keys is not None:
            return
        self._gpu_keys = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=self.config.device, dtype=dtype,
        )
        self._gpu_values = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=self.config.device, dtype=dtype,
        )

        # Pipeline: allocate second buffer pair for ping-pong overlap
        if self._pipeline and self._gpu_keys_alt is None:
            self._gpu_keys_alt = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                device=self.config.device, dtype=dtype,
            )
            self._gpu_values_alt = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                device=self.config.device, dtype=dtype,
            )

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
            store_metadata_on_cpu=self.offload,  # CPU metadata when offloading avoids GPU→CPU sync
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
        Quest mode (paged=True): Quest-selected video pages + text in contiguous 3D buffer.
        """
        cache_layer = self.layers[layer_idx]
        video_len = cache_layer.video_len
        text_seq_len = cache_layer.text_seq_len

        if not self.paged:
            return self._load_layer_continuous(cache_layer, video_len, text_seq_len)
        else:
            keys, values, total_len, _ = self._load_layer_quest(cache_layer, video_len, text_seq_len)
            return keys, values, total_len

    def _load_layer_continuous(
        self,
        cache_layer: SparseCacheLayer,
        video_len: int,
        text_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Non-paged path: reconstruct dense GPU buffer as [video_kv | text_kv]."""
        total_len = video_len + text_seq_len

        # --- Video KV region [0:video_len] ---
        if cache_layer.is_pruned:
            self._fill_pruned_sentinel(self._gpu_keys, self._gpu_values, video_len)

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

    def _load_layer_quest(
        self,
        cache_layer: SparseCacheLayer,
        video_len: int,
        text_seq_len: int,
        target_keys: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
        sync: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Quest + contiguous buffer path.

        When page_min_keys is on CPU (offload mode), Quest scoring runs entirely
        on CPU — avoiding the GPU→CPU sync that GPU-side scoring would require
        to transfer selected page indices back. The Q vector is tiny (a few KB)
        so moving it to CPU is negligible.

        Video KV pages are gathered into a CPU pinned staging buffer before DMA
        to ensure truly asynchronous CPU→GPU transfer (non-pinned source would
        force synchronous memcpy).

        GPU buffer layout:
        - Quest selected (top_k < num_video_pages): [top_k*page_size video tokens | text_seq_len tokens]
        - All kept (top_k >= num_video_pages): [video_len video tokens | text_seq_len tokens]
        - cache_seqlens = video_tokens_used + text_seq_len

        Args:
            target_keys:   GPU buffer to write into (defaults to self._gpu_keys)
            target_values: GPU buffer to write into (defaults to self._gpu_values)
            sync:          If True, synchronize current stream after copy (default).
                           Set False for pipeline async prefetch; caller handles sync.

        Returns:
            (target_keys, target_values, total_len, video_tokens_used)
        """
        if target_keys is None:
            target_keys = self._gpu_keys
        if target_values is None:
            target_values = self._gpu_values

        num_video_pages = cache_layer.num_video_pages
        page_size = self.page_size
        threshold = self.config.dynamic_sparse_threshold

        # === Quest: select Top-K video pages ===
        use_quest = (
            num_video_pages > 0
            and threshold is not None
            and self.current_query is not None
            and cache_layer.page_min_keys is not None
        )
        if use_quest:
            top_k = max(1, min(num_video_pages,
                        int(round(num_video_pages * threshold))))
            if top_k < num_video_pages:
                if cache_layer.page_min_keys.is_cuda:
                    # GPU-side Quest scoring (when metadata is on GPU)
                    q_kv_full = self._map_q_to_kv_heads(self.current_query)
                    if cache_layer.is_pruned:
                        kept_idx = cache_layer.kept_head_indices.to(q_kv_full.device)
                        q_kv = q_kv_full.index_select(0, kept_idx)
                    else:
                        q_kv = q_kv_full
                    q_kv = q_kv.to(dtype=cache_layer.page_min_keys.dtype)
                    scores = cache_layer.quest_score_pages(q_kv)
                    selected_pages = torch.topk(scores, top_k).indices
                    selected_pages, _ = torch.sort(selected_pages)
                    selected_pages = selected_pages.to(device='cpu')
                else:
                    # CPU-side Quest scoring (when metadata is on CPU — avoids GPU→CPU sync)
                    q_kv_full = self._map_q_to_kv_heads(self.current_query)
                    q_kv_cpu = q_kv_full.to(device='cpu', dtype=cache_layer.page_min_keys.dtype)
                    if cache_layer.is_pruned:
                        kept_idx = cache_layer.kept_head_indices
                        q_kv_cpu = q_kv_cpu.index_select(0, kept_idx)
                    scores = cache_layer.quest_score_pages(q_kv_cpu)
                    selected_pages, _ = torch.sort(torch.topk(scores, top_k).indices)
                video_tokens_used = top_k * page_size
            else:
                selected_pages = None
                top_k = num_video_pages
                video_tokens_used = cache_layer.video_len
        else:
            selected_pages = None
            top_k = num_video_pages
            video_tokens_used = cache_layer.video_len

        total_len = video_tokens_used + text_seq_len

        # === Build contiguous video region ===
        if selected_pages is None:
            # All pages kept — single contiguous DMA from pinned memory
            src_k = cache_layer.video_paged_keys
            src_v = cache_layer.video_paged_values

            if cache_layer.is_pruned:
                self._fill_pruned_sentinel(target_keys, target_values, video_tokens_used)
                num_h_stored = src_k.shape[2]
                # Truncate to video_tokens_used to discard zero-padding tokens
                # from the last page. Without this, copy_ fails when video_len
                # is not a multiple of page_size.
                src_k_flat = src_k.reshape(num_video_pages * page_size, num_h_stored, src_k.shape[3])[:video_tokens_used]
                src_v_flat = src_v.reshape(num_video_pages * page_size, num_h_stored, src_v.shape[3])[:video_tokens_used]
                kept = cache_layer.kept_head_indices.tolist()
                for i, h in enumerate(kept):
                    target_keys[:, :video_tokens_used, h:h+1, :].copy_(
                        src_k_flat[:, i:i+1, :], non_blocking=True)
                    target_values[:, :video_tokens_used, h:h+1, :].copy_(
                        src_v_flat[:, i:i+1, :], non_blocking=True)
            else:
                num_h = src_k.shape[2]
                # Truncate to video_tokens_used to discard zero-padding tokens.
                src_k_2d = src_k.reshape(num_video_pages * page_size, num_h, src_k.shape[3])[:video_tokens_used]
                src_v_2d = src_v.reshape(num_video_pages * page_size, num_h, src_v.shape[3])[:video_tokens_used]
                target_keys[:, :video_tokens_used].copy_(src_k_2d, non_blocking=True)
                target_values[:, :video_tokens_used].copy_(src_v_2d, non_blocking=True)
        else:
            # Quest selected pages — DMA each page directly from pinned paged_kv
            # to GPU buffer. Avoids staging buffer double-copy and temporary
            # allocation. Each paged_kv[page_idx] is a view of the pinned tensor,
            # so copy_() is truly async DMA.
            sel_list = selected_pages.tolist()

            if cache_layer.is_pruned:
                self._fill_pruned_sentinel(target_keys, target_values, video_tokens_used)
                kept = cache_layer.kept_head_indices.tolist()
                for i, page_idx in enumerate(sel_list):
                    start = i * page_size
                    end = start + page_size
                    page_k = cache_layer.video_paged_keys[page_idx]  # [page_size, H_stored, D]
                    page_v = cache_layer.video_paged_values[page_idx]
                    for j, h in enumerate(kept):
                        target_keys[:, start:end, h:h+1, :].copy_(
                            page_k[:, j:j+1, :], non_blocking=True)
                        target_values[:, start:end, h:h+1, :].copy_(
                            page_v[:, j:j+1, :], non_blocking=True)
            else:
                for i, page_idx in enumerate(sel_list):
                    start = i * page_size
                    end = start + page_size
                    target_keys[:, start:end].copy_(
                        cache_layer.video_paged_keys[page_idx], non_blocking=True)
                    target_values[:, start:end].copy_(
                        cache_layer.video_paged_values[page_idx], non_blocking=True)

        # === Build contiguous text region ===
        if text_seq_len > 0:
            target_keys[:, video_tokens_used:total_len].copy_(
                cache_layer.text_keys[:, :text_seq_len], non_blocking=True)
            target_values[:, video_tokens_used:total_len].copy_(
                cache_layer.text_values[:, :text_seq_len], non_blocking=True)

        if sync:
            torch.cuda.current_stream().synchronize()
        return target_keys, target_values, total_len, video_tokens_used

    def offload_after_decode(self, layer_idx: int):
        cache_layer = self.layers[layer_idx]

        if not self.paged:
            cache_layer.update_from_gpu_buffer(self._gpu_keys, self._gpu_values)
            return

        # --- Quest contiguous buffer path ---
        # buffer layout: [video_tokens_used | text tokens]
        # new token write position = video_tokens_used + text_seq_len (before incr)
        # need to recompute top_k to determine video_tokens_used
        threshold = self.config.dynamic_sparse_threshold
        num_video_pages = cache_layer.num_video_pages
        if (threshold is not None and num_video_pages > 0
                and cache_layer.page_min_keys is not None):
            top_k = max(1, min(num_video_pages,
                        int(round(num_video_pages * threshold))))
        else:
            top_k = num_video_pages

        if top_k >= num_video_pages:
            # keep all, layout same as continuous mode
            video_tokens_used = cache_layer.video_len
        else:
            video_tokens_used = top_k * self.page_size  # Quest compression

        text_seq_len_before_inc = cache_layer.text_seq_len  # before incr
        buf_pos = video_tokens_used + text_seq_len_before_inc

        cache_layer.text_keys[:, text_seq_len_before_inc:text_seq_len_before_inc + 1].copy_(
            self._gpu_keys[:, buf_pos:buf_pos + 1])
        cache_layer.text_values[:, text_seq_len_before_inc:text_seq_len_before_inc + 1].copy_(
            self._gpu_values[:, buf_pos:buf_pos + 1])
        cache_layer.text_seq_len += 1

    # ----------------------------------------------------------------
    # Pipeline methods: overlap attention computation with KV prefetch
    # ----------------------------------------------------------------

    def _refill_pruned_sentinel(self, layer_idx: int, slot: int, q: torch.Tensor):
        """
        Re-fill pruned head sentinel in a buffer slot with the CORRECT Q.

        In pipeline mode, start_prefetch() fills the sentinel using the PREVIOUS
        layer's Q (an unavoidable approximation — the current layer's Q is not
        yet available when the prefetch happens). This method corrects the
        sentinel on the default stream after the current layer's Q is computed.

        Only touches the video region of pruned head positions; kept heads
        (loaded by the prefetch stream) and text KV are left intact. Values
        remain 0.0 (already set by the prefetch stream's _fill_pruned_sentinel).

        No-op when pruning_k_type != "negative" or the layer has no pruned heads.
        """
        if self.pruning_k_type != "negative":
            return
        cache_layer = self.layers[layer_idx]
        if not cache_layer.is_pruned:
            return

        target_keys, _ = self._get_buffer_pair(slot)
        video_tokens_used = self._slot_info[slot]["video_tokens_used"]
        if video_tokens_used == 0:
            return

        # Compute sentinel using THIS layer's Q
        saved_q = self.current_query
        self.current_query = q
        sentinel = self._get_pruned_key_sentinel()  # [num_kv_heads, head_dim]
        self.current_query = saved_q

        # Index only pruned head positions. Advanced indexing writes a
        # [1, video_tokens_used, num_pruned, D] slice in a single op.
        kept = cache_layer.kept_head_indices
        pruned_mask = torch.ones(self.num_kv_heads, dtype=torch.bool,
                                 device=sentinel.device)
        pruned_mask[kept] = False
        pruned_indices = pruned_mask.nonzero(as_tuple=True)[0]
        if pruned_indices.numel() == 0:
            return

        target_keys[:, :video_tokens_used, pruned_indices, :] = (
            sentinel[pruned_indices][None, None, :, :])

    def enable_pipeline(self):
        """Enable pipeline mode. Must be called before prefill.

        Allocates a dedicated CUDA stream for async DMA prefetch and
        a CUDA event for Q-readiness signaling. Requires Quest mode
        (paged=True), enforced by config validation.
        """
        assert self.paged, "Pipeline requires Quest mode (dynamic_sparse_threshold)"
        self._pipeline = True
        self._prefetch_stream = torch.cuda.Stream()
        self._q_ready_event = torch.cuda.Event(enable_timing=False, blocking=False)

    def _get_buffer_pair(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) GPU buffer for the given slot (0 or 1)."""
        if slot == 0:
            return self._gpu_keys, self._gpu_values
        return self._gpu_keys_alt, self._gpu_values_alt

    def load_layer_to_gpu_pipeline(
        self, layer_idx: int, slot: int, q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load layer KV to a specific buffer slot (synchronous).

        Used for layer 0 (no prior Q for prefetch).

        When q is provided (layer 0 case), it is set as self.current_query
        before loading, so the pruning sentinel uses the correct Q. For layers
        > 0, the buffer is loaded by start_prefetch() with the previous layer's
        Q, then corrected by _refill_pruned_sentinel() on the default stream.

        Args:
            layer_idx: transformer layer index
            slot:      buffer slot (0=primary, 1=alt)
            q:         query tensor for Quest scoring and pruning sentinel.
                       For layer 0 this is the layer's own Q.

        Returns:
            (keys, values, cache_seqlens) for the target buffer slot.
        """
        cache_layer = self.layers[layer_idx]
        video_len = cache_layer.video_len
        text_seq_len = cache_layer.text_seq_len
        target_keys, target_values = self._get_buffer_pair(slot)

        if q is not None:
            self.set_current_query(q)

        _, _, total_len, video_tokens_used = self._load_layer_quest(
            cache_layer, video_len, text_seq_len,
            target_keys=target_keys, target_values=target_values,
            sync=True,
        )

        # Store slot metadata for offload_after_decode_pipeline
        self._slot_info[slot] = {
            "cache_seqlens": total_len,
            "video_tokens_used": video_tokens_used,
        }

        return target_keys, target_values, total_len

    def start_prefetch(self, next_layer_idx: int, q: torch.Tensor, slot: int):
        """Start async prefetch of next layer's KV on the prefetch stream.

        Records a CUDA event after Q computation on the default stream,
        then submits Quest scoring + DMA copy to the prefetch stream.
        Uses stream.wait_event() (GPU-side sync) so the prefetch stream
        waits for Q without blocking the CPU. This allows the CPU to
        immediately submit flash_attn to the default stream.

        The passed Q is the CURRENT layer's Q. For Quest page selection
        this is a standard approximation (similarity of attention patterns
        across adjacent layers). The pruning sentinel filled here uses this
        Q too — _refill_pruned_sentinel() corrects it once the next layer's
        own Q is available.

        Args:
            next_layer_idx: layer to prefetch
            q:              current layer's post-RoPE query [1, 1, H_q, D]
            slot:           target buffer slot for the prefetched data
        """
        # Signal that Q is ready on the default stream
        self._q_ready_event.record(torch.cuda.current_stream())

        # Set Q for Quest scoring (Python attribute, immediate)
        self.set_current_query(q)

        cache_layer = self.layers[next_layer_idx]
        video_len = cache_layer.video_len
        text_seq_len = cache_layer.text_seq_len
        target_keys, target_values = self._get_buffer_pair(slot)

        with torch.cuda.stream(self._prefetch_stream):
            # GPU-side sync: prefetch stream waits for Q to be ready
            # without blocking the CPU (unlike event.synchronize())
            self._prefetch_stream.wait_event(self._q_ready_event)

            _, _, total_len, video_tokens_used = self._load_layer_quest(
                cache_layer, video_len, text_seq_len,
                target_keys=target_keys, target_values=target_values,
                sync=False,  # no sync — caller does wait_prefetch()
            )

            # Store slot metadata for offload_after_decode_pipeline
            self._slot_info[slot] = {
                "cache_seqlens": total_len,
                "video_tokens_used": video_tokens_used,
            }

    def wait_prefetch(self):
        """Block the default stream until the prefetch stream finishes."""
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)

    def offload_after_decode_pipeline(self, layer_idx: int, slot: int):
        """Offload new token's KV from a specific buffer slot back to CacheLayer.

        Reads the buffer layout (video_tokens_used, cache_seqlens) from
        _slot_info[slot] to locate the new token's position in the buffer.

        Args:
            layer_idx: transformer layer index
            slot:      buffer slot that holds this layer's KV
        """
        cache_layer = self.layers[layer_idx]
        target_keys, target_values = self._get_buffer_pair(slot)
        info = self._slot_info[slot]
        video_tokens_used = info["video_tokens_used"]

        # New token is at position video_tokens_used + text_seq_len (before increment)
        text_seq_len_before_inc = cache_layer.text_seq_len
        buf_pos = video_tokens_used + text_seq_len_before_inc

        cache_layer.text_keys[:, text_seq_len_before_inc:text_seq_len_before_inc + 1].copy_(
            target_keys[:, buf_pos:buf_pos + 1])
        cache_layer.text_values[:, text_seq_len_before_inc:text_seq_len_before_inc + 1].copy_(
            target_values[:, buf_pos:buf_pos + 1])
        cache_layer.text_seq_len += 1

    def get_buffer(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (keys, values, cache_seqlens) for a buffer slot.

        Used by the pipeline decode loop to pass the correct buffer
        to flash_attn_with_kvcache.
        """
        keys, values = self._get_buffer_pair(slot)
        cache_seqlens = self._slot_info[slot]["cache_seqlens"]
        return keys, values, cache_seqlens

    def clear(self):
        self.layers = [
            SparseCacheLayer(self.config.max_new_tokens, paged=self.paged, page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]
        # Re-apply head pruning config to freshly created layers.
        self._apply_pruning_info()
        self._gpu_keys = None
        self._gpu_values = None
        self._gpu_keys_alt = None
        self._gpu_values_alt = None
        self.video_info = None
        self.video_indices = None
        self.text_indices = None
        self.current_query = None
        self._slot_info = [
            {"cache_seqlens": 0, "video_tokens_used": 0},
            {"cache_seqlens": 0, "video_tokens_used": 0},
        ]
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
        if self._gpu_keys_alt is not None:
            gpu_buffer += self._gpu_keys_alt.nbytes + self._gpu_values_alt.nbytes

        return {
            "kv_cache_gpu_bytes": kv_gpu,
            "kv_cache_cpu_bytes": kv_cpu,
            "gpu_buffer_bytes": gpu_buffer,
            "video_kv_bytes": video_bytes,
            "text_kv_bytes": text_bytes,
        }

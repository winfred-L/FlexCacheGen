import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo


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

    def __init__(self, max_new_tokens: int):
        # Skip CacheLayer.__init__ to avoid conflict with seq_len property.
        self.max_seq_len: int = 0
        self.max_new_tokens = max_new_tokens
        self.video_keys: torch.Tensor | None = None
        self.video_values: torch.Tensor | None = None
        self.text_keys: torch.Tensor | None = None
        self.text_values: torch.Tensor | None = None
        self.video_len: int = 0
        self.text_seq_len: int = 0

    @property
    def seq_len(self) -> int:
        return self.video_len + self.text_seq_len

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

        # Video storage: exact size, never grows
        self.video_keys = torch.empty(
            (batch_size, video_len, num_heads, head_dim), **alloc_kwargs,
        )
        self.video_values = torch.empty(
            (batch_size, video_len, num_heads, head_dim), **alloc_kwargs,
        )
        self.video_keys.copy_(video_k)
        self.video_values.copy_(video_v)

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
    """

    def __init__(self, config: Config):
        # Skip KVCacheManager.__init__ to use SparseCacheLayer instead
        self.config = config
        self.offload = config.offload_kv_to_cpu
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers

        self.layers: list[SparseCacheLayer] = [
            SparseCacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)
        ]
        self._gpu_keys: torch.Tensor | None = None
        self._gpu_values: torch.Tensor | None = None

        # Video metadata — set by set_video_info() before prefill
        self.video_info: VideoInfo | None = None
        self.video_indices: torch.Tensor | None = None
        self.text_indices: torch.Tensor | None = None

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
        Reconstruct dense GPU buffer as [video_kv | text_kv].
        """
        cache_layer = self.layers[layer_idx]
        video_len = cache_layer.video_len
        text_seq_len = cache_layer.text_seq_len
        total_len = video_len + text_seq_len

        # Copy video KV into buffer [0:video_len]
        self._gpu_keys[:, :video_len].copy_(cache_layer.video_keys, non_blocking=True)
        self._gpu_values[:, :video_len].copy_(cache_layer.video_values, non_blocking=True)

        # Copy text KV into buffer [video_len:video_len+text_seq_len]
        self._gpu_keys[:, video_len:total_len].copy_(
            cache_layer.text_keys[:, :text_seq_len], non_blocking=True
        )
        self._gpu_values[:, video_len:total_len].copy_(
            cache_layer.text_values[:, :text_seq_len], non_blocking=True
        )

        torch.cuda.current_stream().synchronize()

        return self._gpu_keys, self._gpu_values, total_len

    def offload_after_decode(self, layer_idx: int):
        self.layers[layer_idx].update_from_gpu_buffer(self._gpu_keys, self._gpu_values)

    def clear(self):
        self.layers = [
            SparseCacheLayer(self.config.max_new_tokens) for _ in range(self.num_hidden_layers)
        ]
        self._gpu_keys = None
        self._gpu_values = None
        self.video_info = None
        self.video_indices = None
        self.text_indices = None
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, int]:
        """Return KV cache memory usage with video/text breakdown."""
        video_bytes = 0
        text_bytes = 0
        kv_gpu = 0
        kv_cpu = 0
        for layer in self.layers:
            if layer.video_keys is None:
                continue
            v_bytes = layer.video_keys.nbytes + layer.video_values.nbytes
            t_bytes = layer.text_keys.nbytes + layer.text_values.nbytes
            video_bytes += v_bytes
            text_bytes += t_bytes
            layer_bytes = v_bytes + t_bytes
            if layer.video_keys.is_cuda:
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
            "video_kv_bytes": video_bytes,
            "text_kv_bytes": text_bytes,
        }

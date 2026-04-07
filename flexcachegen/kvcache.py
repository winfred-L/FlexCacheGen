import torch
from flexcachegen.config import Config


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

import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo


class CacheLayer:
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    The max capacity is allocated after prefill and limited by `config.max_new_tokens`.
    New decoding KV is updated in flash-attn kernel `flash_attn_with_kvcache`, only `seq_len` needs manually controlled.
    """

    layer_idx: int

    keys: torch.Tensor | None
    values: torch.Tensor | None
    seq_len: int
    max_seq_len: int
    video_info: VideoInfo

    def __init__(self, config: Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.seq_len = 0
        self.max_seq_len = 0
        self.max_new_tokens = self.config.max_new_tokens

        # CPU storage for offloaded KV (pinned memory)
        self.cpu_keys: torch.Tensor | None = None
        self.cpu_values: torch.Tensor | None = None

        # GPU tensors (None when offloaded)
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

        # Sparsity tracking: head indices that were pruned for this layer
        self.pruned_heads: list[int] | None = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        """Initialize cache from prefill k,v tensors.
        - Offload mode: allocate CPU pinned memory, copy GPU->CPU, no GPU tensor kept.
        - Non-offload mode: allocate GPU tensor, copy data.
        """
        dtype = key_states.dtype
        batch_size, seq_len, num_heads, head_dim = key_states.shape
        max_seq_len = seq_len + self.max_new_tokens

        self.seq_len = seq_len
        self.max_seq_len = max_seq_len

        if self.config.offload_kv_to_cpu:
            # Offload mode: allocate CPU pinned buffer, copy from GPU
            self.cpu_keys = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                dtype=dtype, device='cpu',
            ).pin_memory()
            self.cpu_values = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                dtype=dtype, device='cpu',
            ).pin_memory()
            self.cpu_keys[:, :seq_len].copy_(key_states, non_blocking=True)
            self.cpu_values[:, :seq_len].copy_(value_states, non_blocking=True)
            torch.cuda.current_stream().synchronize()

            # No GPU tensor kept
            self.keys = None
            self.values = None
        else:
            # Non-offload mode: allocate GPU tensor
            device = key_states.device
            self.keys = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                device=device, dtype=dtype,
            )
            self.values = torch.empty(
                (batch_size, max_seq_len, num_heads, head_dim),
                device=device, dtype=dtype,
            )
            self.keys[:, :seq_len].copy_(key_states)
            self.values[:, :seq_len].copy_(value_states)

    def load_to_gpu(self, gpu_keys: torch.Tensor, gpu_values: torch.Tensor):
        """
        Load KV cache from CPU pinned memory into provided GPU buffers.
        Pruned heads are already zeroed in CPU data (by apply_pruning_for_kv during prefill),
        so a single bulk copy is sufficient.
        """
        seq_len = self.seq_len

        gpu_keys[:, :seq_len].copy_(self.cpu_keys[:, :seq_len], non_blocking=True)
        gpu_values[:, :seq_len].copy_(self.cpu_values[:, :seq_len], non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # Set temporary GPU reference for decode seq_len tracking
        self.keys = gpu_keys
        self.values = gpu_values

    def offload_to_cpu(self):
        """
        Offload KV cache from GPU to CPU pinned memory, then free GPU tensors.
        Prefill: now a no-op if lazy_initialization already placed data on CPU.
        Decode: only copy the newly appended token.
        """
        # Prefill no-op: lazy_initialization already saved to CPU
        if self.cpu_keys is not None and self.keys is None:
            return

        seq_len = self.seq_len

        if self.cpu_keys is None:
            # Shouldn't happen in normal flow, but handle gracefully
            batch, _, heads, dim = self.keys.shape
            self.cpu_keys = torch.empty(
                (batch, self.max_seq_len, heads, dim),
                dtype=self.keys.dtype, device='cpu',
            ).pin_memory()
            self.cpu_values = torch.empty(
                (batch, self.max_seq_len, heads, dim),
                dtype=self.values.dtype, device='cpu',
            ).pin_memory()
            self.cpu_keys[:, :seq_len].copy_(self.keys[:, :seq_len], non_blocking=True)
            self.cpu_values[:, :seq_len].copy_(self.values[:, :seq_len], non_blocking=True)
        else:
            # Decode: only copy the newly appended token
            pos = seq_len - 1
            self.cpu_keys[:, pos:pos + 1].copy_(self.keys[:, pos:pos + 1], non_blocking=True)
            self.cpu_values[:, pos:pos + 1].copy_(self.values[:, pos:pos + 1], non_blocking=True)

        torch.cuda.current_stream().synchronize()

        # Free GPU tensor references
        self.keys = None
        self.values = None


class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers
        self.num_kv_heads = config.hf_config.text_config.num_key_value_heads

        self._layers = [CacheLayer(self.config, layer_idx) for layer_idx in range(self.num_hidden_layers)]

        # Shared single-layer GPU buffer, reused across layers during decode (lazily allocated)
        self._shared_gpu_keys: torch.Tensor | None = None
        self._shared_gpu_values: torch.Tensor | None = None

    def _ensure_shared_gpu_buffer(self, cache_layer: CacheLayer):
        """Lazily allocate a single-layer-sized GPU buffer for decode-time reuse."""
        if self._shared_gpu_keys is not None:
            return
        batch, _, heads, dim = cache_layer.cpu_keys.shape
        device, dtype = self.config.device, cache_layer.cpu_keys.dtype
        self._shared_gpu_keys = torch.empty(
            (batch, cache_layer.max_seq_len, heads, dim), device=device, dtype=dtype,
        )
        self._shared_gpu_values = torch.empty(
            (batch, cache_layer.max_seq_len, heads, dim), device=device, dtype=dtype,
        )

    def get_shared_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared GPU key/value buffers."""
        return self._shared_gpu_keys, self._shared_gpu_values

    def apply_pruning_for_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply video token pruning directly on provided GPU k, v tensors.
        Zeros out specified heads at video token positions.
        Also stores pruned_heads on the cache layer for later sparse loading.
        Returns k, v (modified in-place).
        """
        cache_layer = self._layers[layer_idx]
        pruning_heads = self.config.pruning_heads

        if pruning_heads is None:
            # No selective pruning config — skip
            return k, v

        if layer_idx not in pruning_heads:
            return k, v

        head_indices = pruning_heads[layer_idx]
        if not head_indices:
            return k, v

        # Store pruned heads for sparse loading
        cache_layer.pruned_heads = head_indices

        # Zero out specified heads at video token positions
        video_info = cache_layer.video_info
        for start, end in video_info.index_ranges:
            k[:, start:end + 1, head_indices, :] = 0
            v[:, start:end + 1, head_indices, :] = 0

        return k, v

    def save_prefill_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Save prefill k,v to cache (CPU if offload enabled, GPU otherwise)."""
        self._layers[layer_idx].lazy_initialization(k, v)

    def _offload_layer_to_cpu(self, layer_idx: int):
        if not self.config.offload_kv_to_cpu:
            return
        self._layers[layer_idx].offload_to_cpu()

    def load_layer_to_gpu(self, layer_idx: int) -> int:
        """Load layer KV to GPU (or reuse existing GPU tensor). Returns cache_seqlens (int)."""
        cache_layer = self._layers[layer_idx]
        if not self.config.offload_kv_to_cpu:
            # Non-offload: data already on GPU, point shared buffer to it
            self._shared_gpu_keys = cache_layer.keys
            self._shared_gpu_values = cache_layer.values
            return cache_layer.seq_len

        self._ensure_shared_gpu_buffer(cache_layer)
        cache_layer.load_to_gpu(self._shared_gpu_keys, self._shared_gpu_values)
        return cache_layer.seq_len

    def update_after_decode(self, layer_idx: int):
        """Increment seq_len after decode token is appended, then offload."""
        self._layers[layer_idx].seq_len += 1
        self._offload_layer_to_cpu(layer_idx)

    def clear(self):
        self._layers = [CacheLayer(self.config, layer_idx) for layer_idx in range(self.num_hidden_layers)]
        self._shared_gpu_keys = None
        self._shared_gpu_values = None

    def set_video_info(self, video_info: VideoInfo):
        for layer_idx in range(self.num_hidden_layers):
            self._layers[layer_idx].video_info = video_info


class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management

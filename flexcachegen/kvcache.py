import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo
# from transformers.cache_utils import DynamicLayer

class CacheLayer:
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    The max capacity is allocated after prefill and limited by `config.max_new_tokens`.
    New decoding KV is updated in flash-attn kernel `flash_attn_with_kvcache`, only `seq_len` needs manually controlled.
    """

    layer_idx: int

    keys: torch.Tensor
    values: torch.Tensor
    seq_len: int
    max_seq_len: int
    video_info: VideoInfo

    def __init__(self, config: Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.seq_len = 0
        self.max_seq_len = 0
        self.max_new_tokens = self.config.max_new_tokens

        # CPU storage for offloaded KV (pinned memory, allocated on first offload)
        self.cpu_keys: torch.Tensor | None = None
        self.cpu_values: torch.Tensor | None = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ):
        device, dtype = key_states.device, key_states.dtype
        batch_size, seq_len, num_heads, head_dim = key_states.shape
        max_seq_len = seq_len + self.max_new_tokens

        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.keys = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.values = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype
        )
        self.keys[:, :seq_len].copy_(key_states)
        self.values[:, :seq_len].copy_(value_states)

    def get_keys(self):
        return self.keys[:, :self.seq_len]

    def get_values(self):
        return self.values[:, :self.seq_len]

    def apply_video_pruning(self):
        """
        Evict video token kv inplace. Do actual compaction (remove token positions), since all heads are pruned.
        Note: `max_seq_len` remains unchanged.
        """
        keep_mask = torch.ones(self.seq_len, dtype=torch.bool, device=self.keys.device)
        for start, end in self.video_info.index_ranges:
            assert start >= 0 and start < self.seq_len, 'video_info.index_ranges.start out of range'
            assert end >= 0 and end < self.seq_len, 'video_info.index_ranges.end out of range'
            keep_mask[start:end+1] = False

        keep_indices = torch.where(keep_mask)[0]
        new_seq_len = keep_indices.shape[0]

        compact_keys = self.keys[:, keep_indices]
        compact_values = self.values[:, keep_indices]

        self.keys[:, :new_seq_len].copy_(compact_keys)
        self.values[:, :new_seq_len].copy_(compact_values)

        self.seq_len = new_seq_len

    def apply_selective_video_pruning(self, head_indices: list[int]):
        """
        Zero out video token KV for specified heads only.
        seq_len remains unchanged (flash_attn requires uniform seq_len across heads).
        """
        for start, end in self.video_info.index_ranges:
            assert start >= 0 and start < self.seq_len, 'video_info.index_ranges.start out of range'
            assert end >= 0 and end < self.seq_len, 'video_info.index_ranges.end out of range'
            self.keys[:, start:end+1, head_indices, :] = 0
            self.values[:, start:end+1, head_indices, :] = 0

    def offload_to_cpu(self):
        """
        Offload KV cache from GPU to CPU pinned memory, then free GPU tensors.
        Prefill (first call): allocate max_seq_len pinned buffer, copy all valid data.
        Decode (subsequent calls): only copy the newly appended token.
        """
        seq_len = self.seq_len

        if self.cpu_keys is None:
            # Prefill: allocate full pinned CPU buffer
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

        # Free GPU tensors (will be replaced by shared buffer on next load)
        self.keys = None
        self.values = None

    def load_to_gpu(self, gpu_keys: torch.Tensor, gpu_values: torch.Tensor):
        """Load KV cache from CPU pinned memory into provided GPU buffers.
        The caller (KVCacheManager) provides pre-allocated shared GPU buffers.
        """
        seq_len = self.seq_len
        gpu_keys[:, :seq_len].copy_(self.cpu_keys[:, :seq_len], non_blocking=True)
        gpu_values[:, :seq_len].copy_(self.cpu_values[:, :seq_len], non_blocking=True)
        torch.cuda.current_stream().synchronize()

        self.keys = gpu_keys
        self.values = gpu_values



class KVCacheManager:
    def __init__(self, config: Config):
        self.config = config
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers
        self.num_kv_heads = config.hf_config.text_config.num_key_value_heads

        self.gpu_buffer = [CacheLayer(self.config, layer_idx) for layer_idx in range(self.num_hidden_layers)]

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

    def offload_layer_to_cpu(self, layer_idx: int):
        if not self.config.offload_kv_to_cpu:
            return
        self.gpu_buffer[layer_idx].offload_to_cpu()

    def load_layer_to_gpu(self, layer_idx: int) -> CacheLayer:
        cache_layer = self.gpu_buffer[layer_idx]
        if not self.config.offload_kv_to_cpu or cache_layer.cpu_keys is None:
            return cache_layer

        self._ensure_shared_gpu_buffer(cache_layer)
        cache_layer.load_to_gpu(self._shared_gpu_keys, self._shared_gpu_values)
        return cache_layer

    def clear(self):
        self.gpu_buffer = [CacheLayer(self.config, layer_idx) for layer_idx in range(self.num_hidden_layers)]
        self._shared_gpu_keys = None
        self._shared_gpu_values = None

    def set_video_info(self, video_info: VideoInfo):
        for layer_idx in range(self.num_hidden_layers):
            self.gpu_buffer[layer_idx].video_info = video_info

    def apply_pruning_for_layer(self, layer_idx: int):
        """
        Apply video token pruning for a specific layer based on config.pruning_heads.
        - pruning_heads is None: prune all heads of all layers (compact, backward compatible)
        - layer_idx not in pruning_heads: skip (no pruning for this layer)
        - all kv heads listed: compact pruning (remove sequence positions)
        - partial heads listed: zero out KV for those heads only
        """
        cache_layer = self.gpu_buffer[layer_idx]
        pruning_heads = self.config.pruning_heads

        if pruning_heads is None:
            cache_layer.apply_video_pruning()
        elif layer_idx not in pruning_heads:
            return
        else:
            head_indices = pruning_heads[layer_idx]
            # if len(head_indices) == self.num_kv_heads:
            #     cache_layer.apply_video_pruning()
            # else:
            #     cache_layer.apply_selective_video_pruning(head_indices)
            cache_layer.apply_selective_video_pruning(head_indices)


class BasicKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement basic kv cache management


class PagedKVCacheManager(KVCacheManager):
    def __init__(self):
        super().__init__()
        # implement paged kv cache management

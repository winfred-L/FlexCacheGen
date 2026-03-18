import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo
from flexcachegen.kernels.sparse_scatter import sparse_scatter


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

        # CPU storage for offloaded KV (pinned memory) — dense path
        self.cpu_keys: torch.Tensor | None = None
        self.cpu_values: torch.Tensor | None = None

        # GPU tensors (None when offloaded)
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

        # Sparsity tracking: head indices that were pruned for this layer
        self.pruned_heads: list[int] | None = None

        # Sparse storage fields
        self.is_sparse: bool = False
        self.cpu_keys_active: torch.Tensor | None = None    # [B, max_S, A, D]
        self.cpu_values_active: torch.Tensor | None = None
        self.cpu_keys_pruned_text: torch.Tensor | None = None   # [B, max_T, P, D]
        self.cpu_values_pruned_text: torch.Tensor | None = None
        self.active_head_indices: list[int] | None = None
        self.pruned_head_indices: list[int] | None = None
        self.text_positions: list[int] | None = None  # non-video seq indices
        self.text_len: int = 0  # current number of text positions stored

        # Cached GPU index tensors (created once, reused every decode step)
        self._active_idx_gpu: torch.Tensor | None = None
        self._pruned_idx_gpu: torch.Tensor | None = None
        # Pinned CPU staging buffer for single-token offload (avoids advanced indexing on GPU)
        self._cpu_token_buf: torch.Tensor | None = None  # [B, 1, H, D]

    def _compute_text_positions(self, seq_len: int) -> list[int]:
        """Compute non-video (text) positions from video_info."""
        video_set = set()
        for start, end in self.video_info.index_ranges:
            for i in range(start, end + 1):
                video_set.add(i)
        return [i for i in range(seq_len) if i not in video_set]

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
            # Check if sparse path should be used
            if self.pruned_heads is not None and len(self.pruned_heads) > 0:
                self._lazy_init_sparse(key_states, value_states, dtype, batch_size, seq_len, num_heads, head_dim, max_seq_len)
            else:
                self._lazy_init_dense(key_states, value_states, dtype, batch_size, seq_len, num_heads, head_dim, max_seq_len)

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

    def _lazy_init_dense(self, key_states, value_states, dtype, batch_size, seq_len, num_heads, head_dim, max_seq_len):
        """Dense offload path: full [B, S, H, D] CPU buffers."""
        self.is_sparse = False
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

    def _lazy_init_sparse(self, key_states, value_states, dtype, batch_size, seq_len, num_heads, head_dim, max_seq_len):
        """Sparse offload path: separate active/pruned head buffers."""
        self.is_sparse = True

        # Compute active vs pruned head indices
        pruned_set = set(self.pruned_heads)
        self.active_head_indices = [h for h in range(num_heads) if h not in pruned_set]
        self.pruned_head_indices = list(self.pruned_heads)

        A = len(self.active_head_indices)
        P = len(self.pruned_head_indices)

        # Compute text positions
        self.text_positions = self._compute_text_positions(seq_len)
        self.text_len = len(self.text_positions)
        max_text_len = self.text_len + self.max_new_tokens  # decode tokens are always text

        # Allocate compact CPU pinned buffers
        self.cpu_keys_active = torch.empty(
            (batch_size, max_seq_len, A, head_dim),
            dtype=dtype, device='cpu',
        ).pin_memory()
        self.cpu_values_active = torch.empty(
            (batch_size, max_seq_len, A, head_dim),
            dtype=dtype, device='cpu',
        ).pin_memory()
        self.cpu_keys_pruned_text = torch.empty(
            (batch_size, max_text_len, P, head_dim),
            dtype=dtype, device='cpu',
        ).pin_memory()
        self.cpu_values_pruned_text = torch.empty(
            (batch_size, max_text_len, P, head_dim),
            dtype=dtype, device='cpu',
        ).pin_memory()

        # Gather active heads from full k,v → cpu_keys_active[:, :seq_len]
        active_idx = torch.tensor(self.active_head_indices, dtype=torch.long, device=key_states.device)
        self.cpu_keys_active[:, :seq_len].copy_(
            key_states[:, :, active_idx, :].contiguous(), non_blocking=True
        )
        self.cpu_values_active[:, :seq_len].copy_(
            value_states[:, :, active_idx, :].contiguous(), non_blocking=True
        )

        # Gather pruned heads at text positions → cpu_keys_pruned_text[:, :text_len]
        if P > 0 and self.text_len > 0:
            pruned_idx = torch.tensor(self.pruned_head_indices, dtype=torch.long, device=key_states.device)
            text_pos_idx = torch.tensor(self.text_positions, dtype=torch.long, device=key_states.device)
            self.cpu_keys_pruned_text[:, :self.text_len].copy_(
                key_states[:, text_pos_idx][:, :, pruned_idx, :].contiguous(), non_blocking=True
            )
            self.cpu_values_pruned_text[:, :self.text_len].copy_(
                value_states[:, text_pos_idx][:, :, pruned_idx, :].contiguous(), non_blocking=True
            )

        torch.cuda.current_stream().synchronize()

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

    def load_to_gpu_sparse(
        self,
        gpu_keys: torch.Tensor,
        gpu_values: torch.Tensor,
        staging_active_keys: torch.Tensor,
        staging_active_values: torch.Tensor,
        staging_pruned_keys: torch.Tensor,
        staging_pruned_values: torch.Tensor,
        head_source: torch.Tensor,
        head_compact_idx: torch.Tensor,
        text_seq_map: torch.Tensor,
    ):
        """
        Load sparse KV cache: DMA compact buffers to GPU staging, then Triton scatter to full layout.
        """
        S = self.seq_len
        A = len(self.active_head_indices)
        P = len(self.pruned_head_indices)
        T = self.text_len

        # CPU buffers are [B, max_S, A, D] with per-layer A — contiguous.
        # Staging buffers are [B, max_S, max_A, D] — slice [:, :S, :A] is non-contiguous when A < max_A.
        # Fix: DMA into a contiguous view by reshaping the flat staging memory.
        B = staging_active_keys.shape[0]
        D = staging_active_keys.shape[3]

        # Use contiguous flat region of staging buffer for active heads
        sa_k = staging_active_keys.reshape(B, -1)[:, :S * A * D].reshape(B, S, A, D)
        sa_v = staging_active_values.reshape(B, -1)[:, :S * A * D].reshape(B, S, A, D)

        # DMA contiguous pinned CPU → contiguous GPU staging
        sa_k.copy_(self.cpu_keys_active[:, :S], non_blocking=True)
        sa_v.copy_(self.cpu_values_active[:, :S], non_blocking=True)

        if P > 0 and T > 0:
            sp_k = staging_pruned_keys.reshape(B, -1)[:, :T * P * D].reshape(B, T, P, D)
            sp_v = staging_pruned_values.reshape(B, -1)[:, :T * P * D].reshape(B, T, P, D)
            sp_k.copy_(self.cpu_keys_pruned_text[:, :T], non_blocking=True)
            sp_v.copy_(self.cpu_values_pruned_text[:, :T], non_blocking=True)
        else:
            sp_k = staging_pruned_keys[:, :1, :max(P, 1)]
            sp_v = staging_pruned_values[:, :1, :max(P, 1)]

        torch.cuda.current_stream().synchronize()

        # Triton scatter into full layout (strides passed through, handles any layout)
        tsm = text_seq_map[:S]
        sparse_scatter(sa_k, sp_k, gpu_keys[:, :S], head_source, head_compact_idx, tsm)
        sparse_scatter(sa_v, sp_v, gpu_values[:, :S], head_source, head_compact_idx, tsm)

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
        if self.is_sparse:
            if self.cpu_keys_active is not None and self.keys is None:
                return
        else:
            if self.cpu_keys is not None and self.keys is None:
                return

        seq_len = self.seq_len

        if self.is_sparse:
            # Sparse decode: new decode tokens are text (non-video), all heads active
            pos = seq_len - 1
            A = len(self.active_head_indices)
            P = len(self.pruned_head_indices)

            # Copy full token GPU→CPU (one contiguous DMA, no gather kernel)
            if self._cpu_token_buf is None:
                B, _, H, D = self.keys.shape
                self._cpu_token_buf = torch.empty(
                    (B, 1, H, D), dtype=self.keys.dtype, device='cpu',
                ).pin_memory()
            self._cpu_token_buf.copy_(self.keys[:, pos:pos + 1], non_blocking=True)
            torch.cuda.current_stream().synchronize()

            # Split on CPU (fast, no GPU overhead)
            self.cpu_keys_active[:, pos, :A].copy_(self._cpu_token_buf[:, 0, self.active_head_indices])
            if P > 0:
                text_pos = self.text_len
                self.cpu_keys_pruned_text[:, text_pos, :P].copy_(self._cpu_token_buf[:, 0, self.pruned_head_indices])

            # Same for values
            self._cpu_token_buf.copy_(self.values[:, pos:pos + 1], non_blocking=True)
            torch.cuda.current_stream().synchronize()

            self.cpu_values_active[:, pos, :A].copy_(self._cpu_token_buf[:, 0, self.active_head_indices])
            if P > 0:
                text_pos = self.text_len
                self.cpu_values_pruned_text[:, text_pos, :P].copy_(self._cpu_token_buf[:, 0, self.pruned_head_indices])

            # Update text_positions list and text_len
            self.text_len += 1
            self.text_positions.append(pos)

            self.keys = None
            self.values = None
            return

        # Dense path
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

        # Shared GPU staging buffers for sparse loading (lazily allocated)
        self._staging_active_keys: torch.Tensor | None = None
        self._staging_active_values: torch.Tensor | None = None
        self._staging_pruned_keys: torch.Tensor | None = None
        self._staging_pruned_values: torch.Tensor | None = None

        # Per-layer GPU metadata tensors for sparse scatter
        self._layer_head_source: dict[int, torch.Tensor] = {}      # layer_idx → [H] int32
        self._layer_head_compact_idx: dict[int, torch.Tensor] = {}  # layer_idx → [H] int32
        self._layer_text_seq_map: dict[int, torch.Tensor] = {}      # layer_idx → [max_S] int32

    def _ensure_shared_gpu_buffer(self, cache_layer: CacheLayer):
        """Lazily allocate a single-layer-sized GPU buffer for decode-time reuse."""
        if self._shared_gpu_keys is not None:
            return

        device, dtype = self.config.device, None

        if cache_layer.is_sparse:
            B, _, A, D = cache_layer.cpu_keys_active.shape
            H = self.num_kv_heads
            dtype = cache_layer.cpu_keys_active.dtype
        else:
            B, _, H, D = cache_layer.cpu_keys.shape
            dtype = cache_layer.cpu_keys.dtype

        self._shared_gpu_keys = torch.empty(
            (B, cache_layer.max_seq_len, H, D), device=device, dtype=dtype,
        )
        self._shared_gpu_values = torch.empty(
            (B, cache_layer.max_seq_len, H, D), device=device, dtype=dtype,
        )

    def _ensure_sparse_staging_buffers(self, cache_layer: CacheLayer):
        """Lazily allocate GPU staging buffers for sparse DMA and pre-compute metadata."""
        if self._staging_active_keys is not None:
            return

        device = self.config.device

        # Find max A and P across all sparse layers for shared staging
        max_A = 0
        max_P = 0
        max_T = 0
        B = D = 0
        dtype = None
        max_seq_len = 0

        for layer in self._layers:
            if layer.is_sparse:
                B_l, _, A_l, D_l = layer.cpu_keys_active.shape
                P_l = len(layer.pruned_head_indices)
                T_l = layer.cpu_keys_pruned_text.shape[1]  # max_text_len allocated
                max_A = max(max_A, A_l)
                max_P = max(max_P, P_l)
                max_T = max(max_T, T_l)
                B = B_l
                D = D_l
                dtype = layer.cpu_keys_active.dtype
                max_seq_len = max(max_seq_len, layer.max_seq_len)

        if max_A == 0:
            return

        self._staging_active_keys = torch.empty(
            (B, max_seq_len, max_A, D), device=device, dtype=dtype,
        )
        self._staging_active_values = torch.empty(
            (B, max_seq_len, max_A, D), device=device, dtype=dtype,
        )
        self._staging_pruned_keys = torch.empty(
            (B, max_T, max(max_P, 1), D), device=device, dtype=dtype,
        )
        self._staging_pruned_values = torch.empty(
            (B, max_T, max(max_P, 1), D), device=device, dtype=dtype,
        )

        # Pre-compute per-layer metadata tensors on GPU
        H = self.num_kv_heads
        for layer in self._layers:
            if not layer.is_sparse:
                continue
            idx = layer.layer_idx

            # head_source: 0=active, 1=pruned
            head_source = torch.zeros(H, dtype=torch.int32, device=device)
            head_compact_idx = torch.zeros(H, dtype=torch.int32, device=device)

            active_map = {}
            for ci, hi in enumerate(layer.active_head_indices):
                active_map[hi] = ci
            pruned_map = {}
            for ci, hi in enumerate(layer.pruned_head_indices):
                pruned_map[hi] = ci

            for h in range(H):
                if h in pruned_map:
                    head_source[h] = 1
                    head_compact_idx[h] = pruned_map[h]
                else:
                    head_source[h] = 0
                    head_compact_idx[h] = active_map[h]

            self._layer_head_source[idx] = head_source
            self._layer_head_compact_idx[idx] = head_compact_idx

            # text_seq_map: seq_pos → text buffer index (-1 for video)
            text_seq_map = torch.full((max_seq_len,), -1, dtype=torch.int32, device=device)
            for ti, sp in enumerate(layer.text_positions):
                text_seq_map[sp] = ti
            self._layer_text_seq_map[idx] = text_seq_map

    def _update_text_seq_map_for_decode(self, layer_idx: int, seq_pos: int, text_idx: int):
        """Update text_seq_map after a decode token is appended (always a text position)."""
        if layer_idx in self._layer_text_seq_map:
            self._layer_text_seq_map[layer_idx][seq_pos] = text_idx

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
        cache_layer = self._layers[layer_idx]
        if cache_layer.is_sparse:
            # Before offloading, record the text_seq_map update for the new decode position
            seq_pos = cache_layer.seq_len - 1  # current seq_len already incremented
            text_idx = cache_layer.text_len  # will be incremented in offload_to_cpu
            self._update_text_seq_map_for_decode(layer_idx, seq_pos, text_idx)
        cache_layer.offload_to_cpu()

    def load_layer_to_gpu(self, layer_idx: int) -> int:
        """Load layer KV to GPU (or reuse existing GPU tensor). Returns cache_seqlens (int)."""
        cache_layer = self._layers[layer_idx]
        if not self.config.offload_kv_to_cpu:
            # Non-offload: data already on GPU, point shared buffer to it
            self._shared_gpu_keys = cache_layer.keys
            self._shared_gpu_values = cache_layer.values
            return cache_layer.seq_len

        self._ensure_shared_gpu_buffer(cache_layer)

        if cache_layer.is_sparse:
            self._ensure_sparse_staging_buffers(cache_layer)
            cache_layer.load_to_gpu_sparse(
                self._shared_gpu_keys,
                self._shared_gpu_values,
                self._staging_active_keys,
                self._staging_active_values,
                self._staging_pruned_keys,
                self._staging_pruned_values,
                self._layer_head_source[layer_idx],
                self._layer_head_compact_idx[layer_idx],
                self._layer_text_seq_map[layer_idx],
            )
        else:
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
        self._staging_active_keys = None
        self._staging_active_values = None
        self._staging_pruned_keys = None
        self._staging_pruned_values = None
        self._layer_head_source.clear()
        self._layer_head_compact_idx.clear()
        self._layer_text_seq_map.clear()

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

import math
import torch
from flexcachegen.config import Config
from flexcachegen.utils import VideoInfo
from flexcachegen.kernels.sparse_scatter import sparse_scatter

class KVReorder:
    _global_instance = None

    def __init__(self, video_info):
        self.original_video_info = video_info

        # prefill_len
        if len(video_info.index_ranges) > 0:
            self.prefill_len = max(end for _, end in video_info.index_ranges) + 1
        else:
            self.prefill_len = 0

        # video positions
        video_positions = []
        for start, end in video_info.index_ranges:
            video_positions.extend(range(start, end + 1))
        video_set = set(video_positions)

        # text positions
        text_positions = [i for i in range(self.prefill_len) if i not in video_set]

        # new order
        self.new_order = video_positions + text_positions

        # inverse order
        self.inverse_order = [0] * self.prefill_len
        for new_idx, old_idx in enumerate(self.new_order):
            self.inverse_order[old_idx] = new_idx

        # new video info
        self.new_video_info = VideoInfo(
            video_info.T_len,
            video_info.H_len,
            video_info.W_len,
            [(0, len(video_positions) - 1)] if video_positions else []
        )

        KVReorder._global_instance = self

    @classmethod
    def get_global(cls):
        if cls._global_instance is None:
            raise RuntimeError("KVReorder not initialized")
        return cls._global_instance

    # reorder
    def reorder(self, k, v):
        if self.prefill_len == 0:
            return k, v

        k_new = k.clone()
        v_new = v.clone()

        k_new[:, :self.prefill_len, :, :] = k[:, self.new_order, :, :]
        v_new[:, :self.prefill_len, :, :] = v[:, self.new_order, :, :]

        return k_new, v_new

    # restore
    def restore(self, k, v):
        if self.prefill_len == 0:
            return k, v

        k_new = k.clone()
        v_new = v.clone()

        k_new[:, :self.prefill_len, :, :] = k[:, self.inverse_order, :, :]
        v_new[:, :self.prefill_len, :, :] = v[:, self.inverse_order, :, :]

        
        return k_new, v_new
    
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

        # reorder
        kv_reorder = KVReorder.get_global()
        key_states, value_states = kv_reorder.reorder(key_states, value_states)

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

        # restore
        kv_reorder = KVReorder.get_global()
        gpu_keys[:, :S], gpu_values[:, :S] = kv_reorder.restore(
            gpu_keys[:, :S],
            gpu_values[:, :S]
        )

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

        # unreorder video info, used for sparse pruning and reordering in KVCacheManager
        self._video_info: VideoInfo | None = None

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

    def get_block_table(self):
        """Return block table for paged attention. None for contiguous mode."""
        return None

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
        
        # use unreorder video info for pruning
        video_info = self._video_info

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

        self._video_info = None

    def set_video_info(self, video_info: VideoInfo):
        self._video_info = video_info
        # initialize
        kv_reorder = KVReorder(video_info)
        # replace video_info
        new_video_info = kv_reorder.new_video_info
        for layer_idx in range(self.num_hidden_layers):
            # self._layers[layer_idx].video_info = video_info
            self._layers[layer_idx].video_info = new_video_info

    def compute_block_importance(self, layer_idx: int, q: torch.Tensor):
        pass


class PagedKVCacheManager:
    """Paged KV cache manager for CPU offload with block-level granularity.

    Stores KV cache in fixed-size blocks on CPU pinned memory. During decode,
    blocks are DMA'd to a GPU block pool and used with flash_attn_with_kvcache's
    block_table parameter. This layout enables future block-level token pruning
    where entire blocks can be skipped during transfer.

    Does NOT inherit from KVCacheManager to avoid inheriting the complex
    sparse/dense dual-path logic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.block_size
        self.num_hidden_layers = config.hf_config.text_config.num_hidden_layers
        self.num_kv_heads = config.hf_config.text_config.num_key_value_heads
        self.head_dim = config.hf_config.text_config.head_dim
        self.dtype = config.dtype
        self.device = config.device

        # Per-layer CPU block storage (pinned memory) — dense path
        # Each: [max_blocks_per_layer, block_size, H, D]
        self._cpu_k_blocks: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._cpu_v_blocks: list[torch.Tensor | None] = [None] * self.num_hidden_layers

        # GPU block pool — single-layer sized, reused across layers during decode
        # Shape: [max_gpu_blocks, block_size, H, D]
        self._gpu_k_pool: torch.Tensor | None = None
        self._gpu_v_pool: torch.Tensor | None = None

        # Block table: logical block index → physical block index (trivially sequential for now)
        self._block_table: list[int] = []
        self._block_table_gpu: torch.Tensor | None = None  # [1, num_used_blocks] int32

        # Sequence tracking
        self._seq_len: int = 0
        self._num_blocks_used: int = 0

        # Sparsity (head-level, applied during prefill but stored densely in blocks)
        self._video_info: VideoInfo | None = None
        self._pruning_heads: dict[int, list[int]] | None = config.pruning_heads

        # Sparse storage (replaces dense blocks when pruning is active)
        self._is_sparse: dict[int, bool] = {}  # per-layer
        self._cpu_keys_active: list[torch.Tensor | None] = [None] * self.num_hidden_layers    # [1, max_S, A, D]
        self._cpu_values_active: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._cpu_keys_pruned_text: list[torch.Tensor | None] = [None] * self.num_hidden_layers  # [1, max_T, P, D]
        self._cpu_values_pruned_text: list[torch.Tensor | None] = [None] * self.num_hidden_layers

        # Per-layer head indices
        self._active_head_indices: dict[int, list[int]] = {}
        self._pruned_head_indices: dict[int, list[int]] = {}
        self._text_positions: list[int] = []
        self._text_len: int = 0

        # GPU staging buffers for sparse DMA (shared across layers, lazily allocated)
        self._staging_active_keys: torch.Tensor | None = None
        self._staging_active_values: torch.Tensor | None = None
        self._staging_pruned_keys: torch.Tensor | None = None
        self._staging_pruned_values: torch.Tensor | None = None

        # Metadata tables for sparse_scatter kernel
        self._layer_head_source: dict[int, torch.Tensor] = {}       # [H] int32
        self._layer_head_compact_idx: dict[int, torch.Tensor] = {}  # [H] int32
        self._text_seq_map: torch.Tensor | None = None               # [max_S] int32

        # CPU staging for decode offload (sparse path)
        self._cpu_token_buf: torch.Tensor | None = None  # [1, 1, H, D]

        # Block-level skip: layer_idx → set of block indices to skip during GPU loading
        self._skipped_blocks: dict[int, set[int]] = {}

        # Block-level importance scoring (extreme K vectors)
        self._block_max_k: list[torch.Tensor | None] = [None] * self.num_hidden_layers  # [num_blocks, H_kv, D]
        self._block_min_k: list[torch.Tensor | None] = [None] * self.num_hidden_layers
        self._block_topk_ratio: float | None = config.block_topk_ratio
        self._num_attention_heads: int = config.hf_config.text_config.num_attention_heads

        # Per-layer selected block indices for compact loading (dense path only)
        self._layer_selected_blocks: dict[int, list[int]] = {}

        # Pipeline state
        self._pipeline_enabled: bool = False
        self._dma_stream: torch.cuda.Stream | None = None

        # Double buffer
        self._gpu_k_pool_A: torch.Tensor | None = None
        self._gpu_v_pool_A: torch.Tensor | None = None
        self._gpu_k_pool_B: torch.Tensor | None = None
        self._gpu_v_pool_B: torch.Tensor | None = None
        self._active_buffer: int = 0  # 0=A active for compute, 1=B

        # Pre-DMA tracking
        self._pre_dma_layer_idx: int | None = None
        self._pre_dma_selected: list[int] | None = None
        self._pre_dma_buffer: int | None = None
        self._pre_dma_event: torch.cuda.Event | None = None
        self._pre_dma_block_table: torch.Tensor | None = None
        self._pre_dma_cache_seqlens: torch.Tensor | None = None

    def set_video_info(self, video_info: VideoInfo):
        # initialize
        kv_reorder = KVReorder(video_info)
        # replace video_info
        self._video_info = kv_reorder.new_video_info
        self._original_video_info = video_info

    def set_skipped_blocks(self, layer_idx: int, block_ids: set[int]):
        """Set which blocks to skip for a given layer during GPU loading."""
        self._skipped_blocks[layer_idx] = block_ids

    def _compute_extreme_k(self, layer_idx: int, k: torch.Tensor):
        """Compute per-block element-wise max and min of K vectors.
        k: [1, S, H_kv, D] on GPU.
        """
        S = k.shape[1]
        H = self.num_kv_heads
        D = self.head_dim
        num_blocks = math.ceil(S / self.block_size)

        max_k = torch.empty(num_blocks, H, D, device=k.device, dtype=k.dtype)
        min_k = torch.empty(num_blocks, H, D, device=k.device, dtype=k.dtype)

        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, S)
            block_k = k[0, start:end]  # [length, H_kv, D]
            max_k[i] = block_k.max(dim=0).values
            min_k[i] = block_k.min(dim=0).values

        self._block_max_k[layer_idx] = max_k
        self._block_min_k[layer_idx] = min_k

    def compute_block_importance(self, layer_idx: int, q: torch.Tensor):
        """Compute per-block importance scores and select top blocks.
        q: [1, 1, H_q, D] on GPU (single decode token).

        Dense layers: store selected block indices for compact loading.
        Sparse layers: fall back to set_skipped_blocks (zeroing).
        """
        if self._block_topk_ratio is None:
            return
        if self._block_max_k[layer_idx] is None:
            return

        if self._is_sparse.get(layer_idx, False):
            # Sparse path: can't compact, fall back to zeroing skipped blocks
            selected_sorted = self._score_blocks(layer_idx, q)
            if selected_sorted is None:
                self._skipped_blocks.pop(layer_idx, None)
                self._layer_selected_blocks.pop(layer_idx, None)
                return
            num_blocks = self._num_blocks_used
            skipped = set(range(num_blocks)) - set(selected_sorted)
            self.set_skipped_blocks(layer_idx, skipped)
            self._layer_selected_blocks.pop(layer_idx, None)
        else:
            # Dense path: use compact block table
            selected_sorted = self._score_blocks(layer_idx, q)
            if selected_sorted is None:
                self._layer_selected_blocks.pop(layer_idx, None)
                return
            self._layer_selected_blocks[layer_idx] = selected_sorted

    def apply_pruning_for_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply video token pruning on GPU k, v tensors (zero out specified heads at video positions).

        Same logic as KVCacheManager.apply_pruning_for_kv — works on dense GPU tensors
        before they are split into paged blocks.
        """
        pruning_heads = self._pruning_heads
        if pruning_heads is None or layer_idx not in pruning_heads:
            return k, v

        head_indices = pruning_heads[layer_idx]
        if not head_indices:
            return k, v

        video_info = self._original_video_info
        for start, end in video_info.index_ranges:
            k[:, start:end + 1, head_indices, :] = 0
            v[:, start:end + 1, head_indices, :] = 0

        return k, v

    def save_prefill_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Save prefill k, v ([B=1, S, H, D] on GPU) into CPU storage."""
        B, S, H, D = k.shape
        pruning_heads = self._pruning_heads
        has_pruning = (
            pruning_heads is not None
            and layer_idx in pruning_heads
            and len(pruning_heads[layer_idx]) > 0
        )

        if has_pruning and self._video_info is not None:
            self._save_prefill_sparse(layer_idx, k, v)
        else:
            self._save_prefill_dense(layer_idx, k, v)

        if self._block_topk_ratio is not None:
            self._compute_extreme_k(layer_idx, k)

    def _save_prefill_dense(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Dense path: store prefill into CPU paged blocks."""
        B, S, H, D = k.shape
        num_blocks = math.ceil(S / self.block_size)

        if self._cpu_k_blocks[layer_idx] is None:
            max_blocks = math.ceil((S + self.config.max_new_tokens) / self.block_size)
            self._cpu_k_blocks[layer_idx] = torch.empty(
                max_blocks, self.block_size, H, D, dtype=k.dtype,
            ).pin_memory()
            self._cpu_v_blocks[layer_idx] = torch.empty(
                max_blocks, self.block_size, H, D, dtype=v.dtype,
            ).pin_memory()

        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, S)
            length = end - start
            self._cpu_k_blocks[layer_idx][i, :length].copy_(k[0, start:end], non_blocking=True)
            self._cpu_v_blocks[layer_idx][i, :length].copy_(v[0, start:end], non_blocking=True)

        torch.cuda.current_stream().synchronize()

        if layer_idx == 0:
            self._seq_len = S
            self._num_blocks_used = num_blocks
            self._block_table = list(range(num_blocks))

    def _compute_text_positions(self, seq_len: int) -> list[int]:
        """Compute non-video (text) positions from video_info."""
        video_set = set()
        for start, end in self._video_info.index_ranges:
            for i in range(start, end + 1):
                video_set.add(i)
        return [i for i in range(seq_len) if i not in video_set]

    def _save_prefill_sparse(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Sparse path: store active/pruned heads separately in compact CPU buffers."""
        
        # reorder
        kv_reorder = KVReorder.get_global()
        k, v = kv_reorder.reorder(k, v)

        B, S, H, D = k.shape
        num_blocks = math.ceil(S / self.block_size)
        max_seq_len = S + self.config.max_new_tokens

        self._is_sparse[layer_idx] = True

        # Compute active vs pruned head indices
        pruned_set = set(self._pruning_heads[layer_idx])
        active_head_indices = [h for h in range(H) if h not in pruned_set]
        pruned_head_indices = list(self._pruning_heads[layer_idx])
        self._active_head_indices[layer_idx] = active_head_indices
        self._pruned_head_indices[layer_idx] = pruned_head_indices

        A = len(active_head_indices)
        P = len(pruned_head_indices)

        # Compute text positions (shared across layers, set once on first sparse layer)
        if not self._text_positions:
            self._text_positions = self._compute_text_positions(S)
            self._text_len = len(self._text_positions)

        text_len = self._text_len
        max_text_len = text_len + self.config.max_new_tokens

        # Allocate compact CPU pinned buffers
        self._cpu_keys_active[layer_idx] = torch.empty(
            (B, max_seq_len, A, D), dtype=k.dtype, device='cpu',
        ).pin_memory()
        self._cpu_values_active[layer_idx] = torch.empty(
            (B, max_seq_len, A, D), dtype=v.dtype, device='cpu',
        ).pin_memory()
        self._cpu_keys_pruned_text[layer_idx] = torch.empty(
            (B, max_text_len, P, D), dtype=k.dtype, device='cpu',
        ).pin_memory()
        self._cpu_values_pruned_text[layer_idx] = torch.empty(
            (B, max_text_len, P, D), dtype=v.dtype, device='cpu',
        ).pin_memory()

        # Gather active heads → cpu_keys_active[:, :S]
        active_idx = torch.tensor(active_head_indices, dtype=torch.long, device=k.device)
        self._cpu_keys_active[layer_idx][:, :S].copy_(
            k[:, :, active_idx, :].contiguous(), non_blocking=True
        )
        self._cpu_values_active[layer_idx][:, :S].copy_(
            v[:, :, active_idx, :].contiguous(), non_blocking=True
        )

        # Gather pruned heads at text positions → cpu_keys_pruned_text[:, :text_len]
        if P > 0 and text_len > 0:
            pruned_idx = torch.tensor(pruned_head_indices, dtype=torch.long, device=k.device)
            text_pos_idx = torch.tensor(self._text_positions, dtype=torch.long, device=k.device)
            self._cpu_keys_pruned_text[layer_idx][:, :text_len].copy_(
                k[:, text_pos_idx][:, :, pruned_idx, :].contiguous(), non_blocking=True
            )
            self._cpu_values_pruned_text[layer_idx][:, :text_len].copy_(
                v[:, text_pos_idx][:, :, pruned_idx, :].contiguous(), non_blocking=True
            )

        torch.cuda.current_stream().synchronize()

        # Block table and seq_len set once on first layer
        if layer_idx == 0:
            self._seq_len = S
            self._num_blocks_used = num_blocks
            self._block_table = list(range(num_blocks))

    def _ensure_gpu_pool(self):
        """Lazily allocate GPU block pool (single-layer sized). Double buffer if pipeline enabled."""
        if self._gpu_k_pool_A is not None:
            return
        max_blocks = math.ceil(
            (self._seq_len + self.config.max_new_tokens) / self.block_size
        )
        shape = (max_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        self._gpu_k_pool_A = torch.empty(*shape, device=self.device, dtype=self.dtype)
        self._gpu_v_pool_A = torch.empty_like(self._gpu_k_pool_A)
        if self._pipeline_enabled:
            self._gpu_k_pool_B = torch.empty(*shape, device=self.device, dtype=self.dtype)
            self._gpu_v_pool_B = torch.empty_like(self._gpu_k_pool_B)
            self._dma_stream = torch.cuda.Stream(device=self.device)
        self._gpu_k_pool = self._gpu_k_pool_A
        self._gpu_v_pool = self._gpu_v_pool_A
        self._update_block_table_gpu()

    def _update_block_table_gpu(self):
        """Sync block table to GPU tensor."""
        self._block_table_gpu = torch.tensor(
            [self._block_table[:self._num_blocks_used]],
            dtype=torch.int32, device=self.device,
        )

    def _ensure_sparse_staging(self):
        """Lazily allocate GPU staging buffers and build metadata for sparse scatter."""
        if self._staging_active_keys is not None:
            return

        device = self.device
        H = self.num_kv_heads

        # Find max A, P, T across all sparse layers
        max_A = 0
        max_P = 0
        max_T = 0
        max_seq_len = 0
        dtype = self.dtype

        for li in range(self.num_hidden_layers):
            if not self._is_sparse.get(li, False):
                continue
            A = len(self._active_head_indices[li])
            P = len(self._pruned_head_indices[li])
            T = self._cpu_keys_pruned_text[li].shape[1]
            max_A = max(max_A, A)
            max_P = max(max_P, P)
            max_T = max(max_T, T)
            max_seq_len = max(max_seq_len, self._cpu_keys_active[li].shape[1])

        if max_A == 0:
            return

        B = 1  # batch size always 1

        self._staging_active_keys = torch.empty(
            (B, max_seq_len, max_A, self.head_dim), device=device, dtype=dtype,
        )
        self._staging_active_values = torch.empty(
            (B, max_seq_len, max_A, self.head_dim), device=device, dtype=dtype,
        )
        self._staging_pruned_keys = torch.empty(
            (B, max_T, max(max_P, 1), self.head_dim), device=device, dtype=dtype,
        )
        self._staging_pruned_values = torch.empty(
            (B, max_T, max(max_P, 1), self.head_dim), device=device, dtype=dtype,
        )

        # Build per-layer metadata tensors
        for li in range(self.num_hidden_layers):
            if not self._is_sparse.get(li, False):
                continue

            head_source = torch.zeros(H, dtype=torch.int32, device=device)
            head_compact_idx = torch.zeros(H, dtype=torch.int32, device=device)

            active_map = {hi: ci for ci, hi in enumerate(self._active_head_indices[li])}
            pruned_map = {hi: ci for ci, hi in enumerate(self._pruned_head_indices[li])}

            for h in range(H):
                if h in pruned_map:
                    head_source[h] = 1
                    head_compact_idx[h] = pruned_map[h]
                else:
                    head_source[h] = 0
                    head_compact_idx[h] = active_map[h]

            self._layer_head_source[li] = head_source
            self._layer_head_compact_idx[li] = head_compact_idx

        # Build shared text_seq_map
        total_max_seq = self._seq_len + self.config.max_new_tokens
        self._text_seq_map = torch.full((total_max_seq,), -1, dtype=torch.int32, device=device)
        for ti, sp in enumerate(self._text_positions):
            self._text_seq_map[sp] = ti

    def enable_pipeline(self):
        """Enable pipeline mode (call after prefill, before decode)."""
        if self.config.block_topk_ratio is None:
            return
        self._pipeline_enabled = True

    def _score_blocks(self, layer_idx: int, q: torch.Tensor) -> list[int] | None:
        """Score blocks and return sorted selected block indices, or None if all needed.

        Extracted from compute_block_importance for reuse in predict_and_start_dma.
        q: [1, 1, H_q, D] on GPU.
        """
        if self._block_topk_ratio is None:
            return None
        if self._block_max_k[layer_idx] is None:
            return None

        num_blocks = self._num_blocks_used
        num_scored = self._block_max_k[layer_idx].shape[0]
        scored_blocks = min(num_blocks, num_scored)

        topk = max(1, int(math.ceil(num_blocks * self._block_topk_ratio)))
        if topk >= num_blocks:
            return None

        H_kv = self.num_kv_heads
        D = self.head_dim
        num_q_per_kv = self._num_attention_heads // H_kv

        q_grouped = q[0, 0].view(H_kv, num_q_per_kv, D)

        max_k = self._block_max_k[layer_idx][:scored_blocks]
        min_k = self._block_min_k[layer_idx][:scored_blocks]

        q_exp = q_grouped.unsqueeze(0)
        max_k_exp = max_k.unsqueeze(2)
        min_k_exp = min_k.unsqueeze(2)

        upper = torch.where(q_exp >= 0, q_exp * max_k_exp, q_exp * min_k_exp)
        scores = upper.sum(dim=-1)
        block_scores = scores.amax(dim=(1, 2))

        if scored_blocks == num_blocks:
            block_scores[-1] = float('inf')

        _, top_indices = block_scores.topk(min(topk, scored_blocks))
        selected = set(top_indices.cpu().tolist())

        for i in range(scored_blocks, num_blocks):
            selected.add(i)
        selected.add(num_blocks - 1)

        selected_sorted = sorted(selected)

        if len(selected_sorted) >= num_blocks:
            return None

        return selected_sorted

    def predict_and_start_dma(self, target_layer_idx: int, predictor_q: torch.Tensor):
        """Use predictor_q (from previous layer) to predict block selection for target_layer_idx,
        and start async DMA to the inactive buffer on the DMA stream."""
        if not self._pipeline_enabled or self._block_topk_ratio is None:
            return
        if self._block_max_k[target_layer_idx] is None:
            return
        if self._is_sparse.get(target_layer_idx, False):
            return

        predicted = self._score_blocks(target_layer_idx, predictor_q)
        if predicted is None:
            return

        target_buf = 1 - self._active_buffer
        tgt_k = self._gpu_k_pool_B if target_buf == 1 else self._gpu_k_pool_A
        tgt_v = self._gpu_v_pool_B if target_buf == 1 else self._gpu_v_pool_A

        # Record event on compute stream, then wait on DMA stream
        wait_event = torch.cuda.Event()
        wait_event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._dma_stream):
            self._dma_stream.wait_event(wait_event)
            for i, blk in enumerate(predicted):
                tgt_k[i].copy_(self._cpu_k_blocks[target_layer_idx][blk], non_blocking=True)
                tgt_v[i].copy_(self._cpu_v_blocks[target_layer_idx][blk], non_blocking=True)
            self._pre_dma_event = torch.cuda.Event()
            self._pre_dma_event.record(self._dma_stream)

        # Pre-build block_table and cache_seqlens
        topk = len(predicted)
        self._pre_dma_block_table = torch.arange(topk, dtype=torch.int32, device=self.device).unsqueeze(0)
        last_block = predicted[-1]
        tokens_in_last = min(self.block_size, self._seq_len - last_block * self.block_size)
        self._pre_dma_cache_seqlens = torch.tensor(
            [(topk - 1) * self.block_size + tokens_in_last], dtype=torch.int32, device=self.device
        )
        self._pre_dma_layer_idx = target_layer_idx
        self._pre_dma_selected = predicted
        self._pre_dma_buffer = target_buf

    def finalize_pre_dma(self, layer_idx: int, actual_q: torch.Tensor):
        """Finalize pre-loaded DMA for layer_idx.

        Compare predicted blocks with actual selection from actual_q.
        Returns (gpu_keys, gpu_values, block_table, cache_seqlens).
        """
        # Get actual block selection
        actual_selected = self._score_blocks(layer_idx, actual_q)

        predicted = self._pre_dma_selected
        target_buf = self._pre_dma_buffer

        tgt_k = self._gpu_k_pool_B if target_buf == 1 else self._gpu_k_pool_A
        tgt_v = self._gpu_v_pool_B if target_buf == 1 else self._gpu_v_pool_A

        if actual_selected is None:
            # All blocks needed — wait for DMA, then do full sync load on active buffer
            if self._pre_dma_event is not None:
                torch.cuda.current_stream(self.device).wait_event(self._pre_dma_event)
            # Fall back: load all blocks synchronously into the current active buffer
            num_blocks = self._num_blocks_used
            self._gpu_k_pool[:num_blocks].copy_(self._cpu_k_blocks[layer_idx][:num_blocks], non_blocking=True)
            self._gpu_v_pool[:num_blocks].copy_(self._cpu_v_blocks[layer_idx][:num_blocks], non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            self._block_table_gpu = torch.tensor(
                [self._block_table[:num_blocks]], dtype=torch.int32, device=self.device
            )
            self._clear_pre_dma_state()
            return (self._gpu_k_pool, self._gpu_v_pool,
                    self._block_table_gpu,
                    torch.tensor([self._seq_len], dtype=torch.int32, device=self.device))

        if actual_selected == predicted:
            # Perfect hit — wait for DMA, switch buffer aliases
            torch.cuda.current_stream(self.device).wait_event(self._pre_dma_event)
            self._active_buffer = target_buf
            self._gpu_k_pool = tgt_k
            self._gpu_v_pool = tgt_v
            block_table = self._pre_dma_block_table
            cache_seqlens = self._pre_dma_cache_seqlens
            # Store selected blocks for update_after_decode
            self._layer_selected_blocks[layer_idx] = actual_selected
            self._clear_pre_dma_state()
            return tgt_k, tgt_v, block_table, cache_seqlens

        # Partial hit — wait for DMA, then fix up
        torch.cuda.current_stream(self.device).wait_event(self._pre_dma_event)

        predicted_set = set(predicted)
        actual_set = set(actual_selected)
        # Blocks already in tgt buffer (at predicted positions)
        predicted_pos = {blk: i for i, blk in enumerate(predicted)}

        # Build actual layout: for each actual block, either reuse from predicted pos or DMA
        topk = len(actual_selected)
        for i, blk in enumerate(actual_selected):
            if blk in predicted_pos:
                src_pos = predicted_pos[blk]
                if src_pos != i:
                    # GPU-internal move within target buffer
                    tgt_k[i].copy_(tgt_k[src_pos])
                    tgt_v[i].copy_(tgt_v[src_pos])
            else:
                # Missing block — sync DMA from CPU
                tgt_k[i].copy_(self._cpu_k_blocks[layer_idx][blk], non_blocking=True)
                tgt_v[i].copy_(self._cpu_v_blocks[layer_idx][blk], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        # Switch to target buffer
        self._active_buffer = target_buf
        self._gpu_k_pool = tgt_k
        self._gpu_v_pool = tgt_v

        block_table = torch.arange(topk, dtype=torch.int32, device=self.device).unsqueeze(0)
        last_block = actual_selected[-1]
        tokens_in_last = min(self.block_size, self._seq_len - last_block * self.block_size)
        cache_seqlens = torch.tensor(
            [(topk - 1) * self.block_size + tokens_in_last], dtype=torch.int32, device=self.device
        )
        self._layer_selected_blocks[layer_idx] = actual_selected
        self._clear_pre_dma_state()
        return tgt_k, tgt_v, block_table, cache_seqlens

    def _clear_pre_dma_state(self):
        """Reset pre-DMA tracking state."""
        self._pre_dma_layer_idx = None
        self._pre_dma_selected = None
        self._pre_dma_buffer = None
        self._pre_dma_event = None
        self._pre_dma_block_table = None
        self._pre_dma_cache_seqlens = None

    def load_layer_to_gpu(self, layer_idx: int) -> int:
        """DMA layer KV from CPU to GPU. Uses sparse or dense path as appropriate.

        Returns cache_seqlens tensor for flash_attn_with_kvcache.
        """
        self._ensure_gpu_pool()

        if self._is_sparse.get(layer_idx, False):
            self._load_layer_sparse(layer_idx)
            # Restore full block table (may have been overwritten by compact load on previous layer)
            self._block_table_gpu = torch.tensor(
                [self._block_table[:self._num_blocks_used]],
                dtype=torch.int32, device=self.device,
            )
            return torch.tensor([self._seq_len], dtype=torch.int32, device=self.device)

        # Dense path
        selected = self._layer_selected_blocks.get(layer_idx)
        if selected is not None:
            return self._load_layer_compact(layer_idx, selected)
        else:
            num_blocks = self._num_blocks_used
            self._gpu_k_pool[:num_blocks].copy_(self._cpu_k_blocks[layer_idx][:num_blocks], non_blocking=True)
            self._gpu_v_pool[:num_blocks].copy_(self._cpu_v_blocks[layer_idx][:num_blocks], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            # Restore full block table (may have been overwritten by compact load on previous layer)
            self._block_table_gpu = torch.tensor(
                [self._block_table[:self._num_blocks_used]],
                dtype=torch.int32, device=self.device,
            )
            return torch.tensor([self._seq_len], dtype=torch.int32, device=self.device)

    def _load_layer_compact(self, layer_idx: int, selected: list[int]) -> torch.Tensor:
        """Load only selected blocks into contiguous GPU positions.

        Returns compact cache_seqlens.  Attention only sees selected blocks,
        reducing both DMA and attention computation.
        """
        topk = len(selected)

        # Batch DMA: copy selected CPU blocks → GPU pool positions 0..topk-1
        for i, block_idx in enumerate(selected):
            self._gpu_k_pool[i].copy_(self._cpu_k_blocks[layer_idx][block_idx], non_blocking=True)
            self._gpu_v_pool[i].copy_(self._cpu_v_blocks[layer_idx][block_idx], non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # Compact block table: identity mapping [0, 1, ..., topk-1]
        self._block_table_gpu = torch.arange(
            topk, dtype=torch.int32, device=self.device,
        ).unsqueeze(0)

        # Compact seq_len: (topk-1) full blocks + tokens in last selected block
        last_block = selected[-1]
        tokens_in_last = min(self.block_size, self._seq_len - last_block * self.block_size)
        compact_seqlen = (topk - 1) * self.block_size + tokens_in_last

        return torch.tensor([compact_seqlen], dtype=torch.int32, device=self.device)

    def _load_layer_sparse(self, layer_idx: int):
        """Sparse load: DMA compact buffers to GPU staging, then Triton scatter into GPU block pool."""
        self._ensure_sparse_staging()

        S = self._seq_len
        A = len(self._active_head_indices[layer_idx])
        P = len(self._pruned_head_indices[layer_idx])
        T = self._text_len
        H = self.num_kv_heads
        D = self.head_dim
        num_blocks = self._num_blocks_used
        B = 1

        # DMA active heads: use contiguous flat region of staging buffer
        sa_k = self._staging_active_keys.reshape(B, -1)[:, :S * A * D].reshape(B, S, A, D)
        sa_v = self._staging_active_values.reshape(B, -1)[:, :S * A * D].reshape(B, S, A, D)
        sa_k.copy_(self._cpu_keys_active[layer_idx][:, :S], non_blocking=True)
        sa_v.copy_(self._cpu_values_active[layer_idx][:, :S], non_blocking=True)

        # DMA pruned heads at text positions
        if P > 0 and T > 0:
            sp_k = self._staging_pruned_keys.reshape(B, -1)[:, :T * P * D].reshape(B, T, P, D)
            sp_v = self._staging_pruned_values.reshape(B, -1)[:, :T * P * D].reshape(B, T, P, D)
            sp_k.copy_(self._cpu_keys_pruned_text[layer_idx][:, :T], non_blocking=True)
            sp_v.copy_(self._cpu_values_pruned_text[layer_idx][:, :T], non_blocking=True)
        else:
            sp_k = self._staging_pruned_keys[:, :1, :max(P, 1)]
            sp_v = self._staging_pruned_values[:, :1, :max(P, 1)]

        torch.cuda.current_stream().synchronize()

        # Reshape GPU block pool to [1, num_blocks*block_size, H, D] for scatter
        full_k = self._gpu_k_pool[:num_blocks].reshape(1, num_blocks * self.block_size, H, D)
        full_v = self._gpu_v_pool[:num_blocks].reshape(1, num_blocks * self.block_size, H, D)

        tsm = self._text_seq_map[:S]
        sparse_scatter(sa_k, sp_k, full_k[:, :S], self._layer_head_source[layer_idx], self._layer_head_compact_idx[layer_idx], tsm)
        sparse_scatter(sa_v, sp_v, full_v[:, :S], self._layer_head_source[layer_idx], self._layer_head_compact_idx[layer_idx], tsm)

        # restore
        kv_reorder = KVReorder.get_global()
        full_k[:, :S], full_v[:, :S] = kv_reorder.restore(
            full_k[:, :S],
            full_v[:, :S]
        )

        # Zero out skipped blocks after scatter
        skipped = self._skipped_blocks.get(layer_idx)
        if skipped:
            for bid in skipped:
                if bid < num_blocks:
                    self._gpu_k_pool[bid].zero_()
                    self._gpu_v_pool[bid].zero_()

    def get_shared_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GPU block pool tensors [max_blocks, block_size, H, D]."""
        return self._gpu_k_pool, self._gpu_v_pool

    def get_block_table(self) -> torch.Tensor:
        """Return GPU block table [1, num_used_blocks] int32."""
        return self._block_table_gpu

    def update_after_decode(self, layer_idx: int):
        """After flash_attn_with_kvcache writes the new token into GPU block pool,
        copy that single slot back to CPU and manage block allocation."""
        new_seq_len = self._seq_len + 1

        if self._is_sparse.get(layer_idx, False):
            self._update_after_decode_sparse(layer_idx, new_seq_len)
        else:
            self._update_after_decode_dense(layer_idx, new_seq_len)

        # Update seq_len and block allocation on the last layer
        if layer_idx == self.num_hidden_layers - 1:
            torch.cuda.current_stream().synchronize()
            self._seq_len = new_seq_len
            if new_seq_len % self.block_size == 0:
                new_block_id = self._num_blocks_used
                self._block_table.append(new_block_id)
                self._num_blocks_used += 1
                self._update_block_table_gpu()

    def _update_after_decode_dense(self, layer_idx: int, new_seq_len: int):
        """Dense path: copy single slot from GPU pool back to CPU block."""
        pos_in_block = (new_seq_len - 1) % self.block_size
        orig_block_idx = (new_seq_len - 1) // self.block_size

        selected = self._layer_selected_blocks.get(layer_idx)
        if selected is not None:
            # Compact mode: new token is in the last compact GPU block
            gpu_block = len(selected) - 1
        else:
            gpu_block = self._block_table[orig_block_idx]

        self._cpu_k_blocks[layer_idx][orig_block_idx, pos_in_block].copy_(
            self._gpu_k_pool[gpu_block, pos_in_block], non_blocking=True
        )
        self._cpu_v_blocks[layer_idx][orig_block_idx, pos_in_block].copy_(
            self._gpu_v_pool[gpu_block, pos_in_block], non_blocking=True
        )

    def _update_after_decode_sparse(self, layer_idx: int, new_seq_len: int):
        """Sparse path: copy full token GPU→CPU, split into active/pruned buffers."""
        pos = new_seq_len - 1
        H = self.num_kv_heads
        D = self.head_dim

        # Lazily allocate CPU staging buffer
        if self._cpu_token_buf is None:
            self._cpu_token_buf = torch.empty(
                (1, 1, H, D), dtype=self.dtype, device='cpu',
            ).pin_memory()

        # Read new token from GPU block pool
        block_idx = pos // self.block_size
        pos_in_block = pos % self.block_size
        phys_block = self._block_table[block_idx]

        active_indices = self._active_head_indices[layer_idx]
        pruned_indices = self._pruned_head_indices[layer_idx]
        A = len(active_indices)
        P = len(pruned_indices)

        # Keys: GPU → CPU staging → split
        self._cpu_token_buf[0, 0].copy_(self._gpu_k_pool[phys_block, pos_in_block], non_blocking=True)
        torch.cuda.current_stream().synchronize()

        self._cpu_keys_active[layer_idx][:, pos, :A].copy_(self._cpu_token_buf[:, 0, active_indices])
        if P > 0:
            text_pos = self._text_len
            self._cpu_keys_pruned_text[layer_idx][:, text_pos, :P].copy_(self._cpu_token_buf[:, 0, pruned_indices])

        # Values: GPU → CPU staging → split
        self._cpu_token_buf[0, 0].copy_(self._gpu_v_pool[phys_block, pos_in_block], non_blocking=True)
        torch.cuda.current_stream().synchronize()

        self._cpu_values_active[layer_idx][:, pos, :A].copy_(self._cpu_token_buf[:, 0, active_indices])
        if P > 0:
            text_pos = self._text_len
            self._cpu_values_pruned_text[layer_idx][:, text_pos, :P].copy_(self._cpu_token_buf[:, 0, pruned_indices])

        # Update text_seq_map and text tracking on last layer
        if layer_idx == self.num_hidden_layers - 1:
            if self._text_seq_map is not None:
                self._text_seq_map[pos] = self._text_len
            self._text_len += 1
            self._text_positions.append(pos)

    def clear(self):
        """Reset all state, release CPU/GPU buffers."""
        self._cpu_k_blocks = [None] * self.num_hidden_layers
        self._cpu_v_blocks = [None] * self.num_hidden_layers
        self._gpu_k_pool = None
        self._gpu_v_pool = None
        self._block_table = []
        self._block_table_gpu = None
        self._seq_len = 0
        self._num_blocks_used = 0
        self._video_info = None

        # Sparse state
        self._is_sparse.clear()
        self._cpu_keys_active = [None] * self.num_hidden_layers
        self._cpu_values_active = [None] * self.num_hidden_layers
        self._cpu_keys_pruned_text = [None] * self.num_hidden_layers
        self._cpu_values_pruned_text = [None] * self.num_hidden_layers
        self._active_head_indices.clear()
        self._pruned_head_indices.clear()
        self._text_positions = []
        self._text_len = 0
        self._staging_active_keys = None
        self._staging_active_values = None
        self._staging_pruned_keys = None
        self._staging_pruned_values = None
        self._layer_head_source.clear()
        self._layer_head_compact_idx.clear()
        self._text_seq_map = None
        self._cpu_token_buf = None
        self._skipped_blocks.clear()
        self._block_max_k = [None] * self.num_hidden_layers
        self._block_min_k = [None] * self.num_hidden_layers
        self._layer_selected_blocks.clear()

        # Pipeline state
        self._pipeline_enabled = False
        self._dma_stream = None
        self._gpu_k_pool_A = None
        self._gpu_v_pool_A = None
        self._gpu_k_pool_B = None
        self._gpu_v_pool_B = None
        self._active_buffer = 0
        self._clear_pre_dma_state()

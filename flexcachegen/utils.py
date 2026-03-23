import torch
from time import perf_counter
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass


@contextmanager
def nvtx_range(name: str, enabled: bool = True):
    """NVTX range context manager for nsys profiling. No-op when disabled."""
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()

@dataclass
class VideoInfo:
    T_len: int
    H_len: int
    W_len: int
    index_ranges: list[tuple[int, int]]
        # list of (start_idx, end_idx) of input video token, aka <|video_pad|>, end_idx is included



def get_tensor_size(tensor: torch.Tensor) -> str:
    num_elements = tensor.nelement()
    element_size = tensor.element_size()
    total_bytes = num_elements * element_size

    if total_bytes < 1024:
        return f"{total_bytes} B"
    elif total_bytes < 1024 ** 2:
        return f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024 ** 3:
        return f"{total_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{total_bytes / (1024 ** 3):.2f} GB"
    

def print_cuda_memory_usage(device: torch.device):
    cur_mem = torch.cuda.memory_allocated(device) / 1024 ** 3
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(f"Memory Usage: Current {cur_mem:.2f} GB, Peak {peak_mem:.2f} GB")


def print_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - t
        print(f"[{func.__name__}] Duration: {duration:.2f} seconds")
        return result
    return wrapper
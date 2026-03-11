import torch
from time import perf_counter
from functools import wraps

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
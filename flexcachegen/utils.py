import torch
from time import perf_counter

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
    cur_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f"Current memory allocated: {cur_mem / 1024 ** 3:.2f} GB")
    print(f"Peak memory allocated: {peak_mem / 1024 ** 3:.2f} GB")


def print_duration(start, end):
    duration = end - start
    print(f"Duration: {duration:.2f} seconds")
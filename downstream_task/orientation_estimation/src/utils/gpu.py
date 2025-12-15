import torch
def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
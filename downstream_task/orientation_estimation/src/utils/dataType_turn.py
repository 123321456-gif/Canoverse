
import torch
from pathlib import Path


def torch_tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.tolist()

def list_to_torch_tensor(lst: list, device='cpu') -> torch.Tensor:
    return torch.tensor(lst).to(device)

def path_to_str(path: Path) -> str:
    return str(path)

def str_to_path(path_str: str) -> Path:
    return Path(path_str)
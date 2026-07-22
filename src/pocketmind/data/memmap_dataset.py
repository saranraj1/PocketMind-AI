import numpy as np
import torch
from torch.utils.data import Dataset

class MemMapDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int, dtype=np.uint16):
        self.bin_path = bin_path
        self.context_length = context_length
        self.dtype = dtype
        
        # Use np.memmap for memory-efficient access
        self.tokens = np.memmap(bin_path, dtype=dtype, mode="r")
        self.num_tokens = len(self.tokens)
        
        # We need context_length + 1 tokens to construct (x, y) target-shifted inputs
        self.length = max(0, self.num_tokens - context_length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds for MemMapDataset")
            
        chunk = self.tokens[idx : idx + self.context_length + 1]
        
        # Convert to torch tensors
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y

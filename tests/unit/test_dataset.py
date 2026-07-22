import os
import tempfile
import numpy as np
import pytest
import torch
from pocketmind.data.memmap_dataset import MemMapDataset

def test_memmap_dataset_shifts():
    # Create a temporary binary file with mock token IDs [0, 1, 2, ..., 9]
    tokens = np.array(list(range(10)), dtype=np.uint16)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(tokens.tobytes())
        tmp_name = tmp.name
        
    try:
        context_length = 4
        dataset = MemMapDataset(tmp_name, context_length=context_length, dtype=np.uint16)
        
        # Total tokens = 10, context_length = 4
        # Expected len = 10 - 4 = 6
        assert len(dataset) == 6
        
        # Verify first item (idx = 0)
        x, y = dataset[0]
        # x should be [0, 1, 2, 3]
        # y should be [1, 2, 3, 4]
        assert torch.equal(x, torch.tensor([0, 1, 2, 3], dtype=torch.long))
        assert torch.equal(y, torch.tensor([1, 2, 3, 4], dtype=torch.long))
        
        # Test required invariant: x[1:] equals y[:-1]
        assert torch.equal(x[1:], y[:-1])
        
        # Verify last item (idx = 5)
        x_last, y_last = dataset[5]
        # x_last should be [5, 6, 7, 8]
        # y_last should be [6, 7, 8, 9]
        assert torch.equal(x_last, torch.tensor([5, 6, 7, 8], dtype=torch.long))
        assert torch.equal(y_last, torch.tensor([6, 7, 8, 9], dtype=torch.long))
        assert torch.equal(x_last[1:], y_last[:-1])
        
        # Out of bounds should raise IndexError
        with pytest.raises(IndexError):
            _ = dataset[6]
            
    finally:
        if "dataset" in locals() and hasattr(dataset, "tokens") and hasattr(dataset.tokens, "_mmap"):
            try:
                dataset.tokens._mmap.close()
            except Exception:
                pass
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception as e:
                print(f"Warning: could not remove temp file {tmp_name}: {e}")

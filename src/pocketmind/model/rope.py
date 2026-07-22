import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # inv_freq configuration (applied to even index elements)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.cos_cached = None
        self.sin_cached = None

    def _update_cache(self, x: torch.Tensor, seq_len: int) -> None:
        device = x.device
        dtype = x.dtype
        
        # Check if the cache is already allocated and sufficiently long
        if (
            self.cos_cached is not None 
            and self.cos_cached.shape[0] >= seq_len 
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return
            
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Concatenate freqs to form full rotation matrix inputs
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, H, head_dim]
        # Return cos and sin sliced to seq_len of shape [1, T, 1, head_dim]
        self._update_cache(x, seq_len)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)
        return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Rotate query or key tensor x [B, T, H, D] using cached cos/sin [1, T, 1, D]
    d = x.shape[-1]
    x_rot = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    return x * cos + x_rot * sin

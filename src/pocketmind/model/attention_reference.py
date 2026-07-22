import math
import torch
import torch.nn as nn
from pocketmind.config import PocketMindConfig
from pocketmind.model.rope import apply_rope

class ReferenceCausalAttention(nn.Module):
    def __init__(self, config: PocketMindConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = max(1, min(config.n_kv_heads, config.n_heads))
        self.head_dim = config.d_model // config.n_heads
        self.attention_window = config.attention_window
        
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        
        # Projections
        self.wq = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # x shape: [B, T, D]
        B, T, _ = x.shape
        
        # Project inputs
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Determine position indices for RoPE based on cache size
        start_pos = 0
        if kv_cache is not None:
            start_pos = kv_cache[0].shape[1]
            
        # Slice cos/sin to match current sequence positions
        cos_sliced = cos[:, start_pos : start_pos + T]
        sin_sliced = sin[:, start_pos : start_pos + T]
        
        # Apply RoPE
        q = apply_rope(q, cos_sliced, sin_sliced)
        k = apply_rope(k, cos_sliced, sin_sliced)
        
        # Concatenate keys/values with past KV Cache
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)
        new_kv_cache = (k, v)
            
        # Repeat key/value heads to match query heads (Grouped Query Attention)
        if self.num_queries_per_kv > 1:
            k_rep = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v_rep = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        else:
            k_rep = k
            v_rep = v
            
        # Transpose to shape [B, H, T, head_dim] for attention computation
        q = q.transpose(1, 2)              # [B, H, T_q, head_dim]
        k_rep = k_rep.transpose(1, 2)      # [B, H, T_k, head_dim]
        v_rep = v_rep.transpose(1, 2)      # [B, H, T_v, head_dim]
        
        # Compute raw attention scores
        # scores: [B, H, T_q, T_k]
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Build local causal attention mask
        # Columns correspond to k index (T_k), rows to q index (T_q)
        T_q = T
        T_k = k.shape[1]
        
        # Initialize float mask with zeros
        mask = torch.zeros((T_q, T_k), device=x.device, dtype=x.dtype)
        
        # Local Causal constraints
        q_idx = torch.arange(start_pos, start_pos + T_q, device=x.device).view(-1, 1)
        k_idx = torch.arange(T_k, device=x.device).view(1, -1)
        
        # 1) Causal: k_idx <= q_idx
        # 2) Local window: q_idx - k_idx < attention_window
        causal_ok = (k_idx <= q_idx)
        local_ok = (q_idx - k_idx < self.attention_window)
        valid_attn = causal_ok & local_ok
        
        # Mask out invalid locations with -inf
        mask = mask.masked_fill(~valid_attn, float("-inf"))
        
        # Add mask to scores (broadcasting across B and H)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)
        
        # Softmax & Dropout
        probs = torch.softmax(scores.float(), dim=-1).type_as(scores)
        probs = self.dropout(probs)
        
        # Weighted values
        # output: [B, H, T_q, head_dim]
        output = torch.matmul(probs, v_rep)
        
        # Re-assemble output heads
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.wo(output), new_kv_cache

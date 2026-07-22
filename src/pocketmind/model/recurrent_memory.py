import torch
import torch.nn as nn
from pocketmind.config import PocketMindConfig

class GatedRecurrentMemory(nn.Module):
    def __init__(self, config: PocketMindConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.recurrent_state_dim = config.recurrent_state_dim
        
        # Projections for gate
        self.w_gate = nn.Linear(self.d_model, self.recurrent_state_dim, bias=True)
        self.u_gate = nn.Linear(self.recurrent_state_dim, self.recurrent_state_dim, bias=False)
        
        # Projections for candidate representation
        self.w_in = nn.Linear(self.d_model, self.recurrent_state_dim, bias=True)
        self.w_state = nn.Linear(self.recurrent_state_dim, self.recurrent_state_dim, bias=False)
        
        # Output projection back to model dimension
        self.w_out = nn.Linear(self.recurrent_state_dim, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        kv_cache: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, d_model]
        # kv_cache is the past recurrent state of shape [B, recurrent_state_dim]
        B, T, _ = x.shape
        
        # Initialize state to zeros if not provided
        if kv_cache is not None:
            state = kv_cache
        else:
            state = torch.zeros(
                B, self.recurrent_state_dim,
                device=x.device,
                dtype=x.dtype
            )
            
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            
            # gate = sigmoid(x_t @ W_gate + state @ U_gate + b_gate)
            gate = torch.sigmoid(self.w_gate(x_t) + self.u_gate(state))
            
            # candidate = tanh(x_t @ W_in + state @ W_state + b_state)
            candidate = torch.tanh(self.w_in(x_t) + self.w_state(state))
            
            # state update: gate * state + (1 - gate) * candidate
            state = gate * state + (1.0 - gate) * candidate
            
            # Project back to model dimension width
            out_t = self.w_out(state)
            outputs.append(out_t)
            
        # Reconstruct output sequence
        output = torch.stack(outputs, dim=1)
        
        return output, state

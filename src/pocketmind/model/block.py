import torch
import torch.nn as nn
from pocketmind.model.rmsnorm import RMSNorm

class PocketMindBlock(nn.Module):
    def __init__(self, sequence_mixer: nn.Module, feed_forward: nn.Module, d_model: int):
        super().__init__()
        self.sequence_mixer = sequence_mixer
        self.feed_forward = feed_forward
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # Pre-normalization residual layout
        # 1) Normalization -> sequence mixer (Attention or Recurrent memory) -> Residual
        normed_x = self.norm1(x)
        mixer_outputs = self.sequence_mixer(normed_x, *args, **kwargs)
        
        if isinstance(mixer_outputs, tuple):
            mixer_out, extra_state = mixer_outputs
        else:
            mixer_out, extra_state = mixer_outputs, None
            
        x = x + mixer_out
        
        # 2) Normalization -> SwiGLU FFN -> Residual
        x = x + self.feed_forward(self.norm2(x))
        
        return x, extra_state

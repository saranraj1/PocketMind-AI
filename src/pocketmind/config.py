from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

class PocketMindConfig(BaseModel):
    # Model parameters
    vocab_size: int = Field(default=8000, description="Size of tokenizer vocabulary")
    context_length: int = Field(default=512, description="Maximum sequence length")
    d_model: int = Field(default=512, description="Model dimension width")
    n_layers: int = Field(default=16, description="Number of sequence blocks")
    n_heads: int = Field(default=8, description="Number of attention heads")
    n_kv_heads: int = Field(default=4, description="Number of key/value heads for GQA")
    ffn_dim: int = Field(default=2048, description="Intermediate SwiGLU FFN dimension")
    dropout: float = Field(default=0.10, description="Dropout rate")
    attention_window: int = Field(default=128, description="Local attention window range")
    recurrent_state_dim: int = Field(default=512, description="Width of recurrent memory state")
    tie_embeddings: bool = Field(default=True, description="Whether to share embedding and LM head weights")
    norm: Literal["rmsnorm"] = Field(default="rmsnorm", description="Type of normalization layer")
    activation: Literal["silu"] = Field(default="silu", description="Activation function")
    position_encoding: Literal["rope"] = Field(default="rope", description="Position representation encoding")
    use_optimized_attention: bool = Field(default=True, description="Whether to use PyTorch fused scaled dot-product attention")
    
    # Training parameters
    micro_batch_size: int = Field(default=4, description="Sequence examples per forward step")
    gradient_accumulation: int = Field(default=16, description="Accumulation steps before optimizer step")
    max_steps: int = Field(default=50000, description="Maximum training steps")
    learning_rate: float = Field(default=3e-4, description="Initial learning rate")
    weight_decay: float = Field(default=0.10, description="Optimizer weight decay coefficient")
    activation_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    precision: Literal["fp32", "fp16", "bf16"] = Field(default="fp16", description="Floating-point precision setting")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PocketMindConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

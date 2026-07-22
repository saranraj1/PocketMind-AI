import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pocketmind.config import PocketMindConfig
from pocketmind.model.rmsnorm import RMSNorm
from pocketmind.model.rope import RoPE
from pocketmind.model.attention_reference import ReferenceCausalAttention
from pocketmind.model.attention_optimized import OptimizedCausalAttention
from pocketmind.model.recurrent_memory import GatedRecurrentMemory
from pocketmind.model.feed_forward import SwiGLUFFN
from pocketmind.model.block import PocketMindBlock

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    retrieval_embedding: torch.Tensor | None = None
    recurrent_state: list[torch.Tensor] | None = None
    kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)

class PocketMindModel(nn.Module):
    def __init__(self, config: PocketMindConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional Encoding (RoPE)
        head_dim = config.d_model // config.n_heads
        self.rope = RoPE(dim=head_dim, max_seq_len=config.context_length)
        
        # Transformer blocks (All local attention by default for Phase 2 baseline)
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            # Check if this layer is recurrent (3rd block of each sequence group)
            is_recurrent = (i in {2, 5, 9}) if config.n_layers == 12 else ((i + 1) % 3 == 0)
            
            if is_recurrent:
                mixer = GatedRecurrentMemory(config)
            else:
                mixer = (
                    OptimizedCausalAttention(config)
                    if config.use_optimized_attention
                    else ReferenceCausalAttention(config)
                )
                
            self.blocks.append(
                PocketMindBlock(
                    sequence_mixer=mixer,
                    feed_forward=SwiGLUFFN(config.d_model, config.ffn_dim, config.dropout),
                    d_model=config.d_model
                )
            )
        
        # Final pre-norm layout normalization
        self.norm = RMSNorm(config.d_model)
        
        # Output language modeling projection head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
            
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        memory_ids: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ) -> ModelOutput:
        # B: batch size, T: sequence length
        B, T = input_ids.shape
        
        # Embed tokens
        h = self.embeddings(input_ids)
        
        # Determine total sequence length for RoPE caching (including past KV Cache size)
        total_len = T
        if kv_caches is not None:
            # Find the size of the existing attention cache
            for cache in kv_caches:
                if cache is not None and isinstance(cache, tuple):
                    total_len = T + cache[0].shape[1]
                    break
                    
        # Compute RoPE sine and cosine embeddings
        cos, sin = self.rope(h, total_len)
        
        # Feed through transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            
            # Forward pass of block
            h, updated_cache = block(h, cos=cos, sin=sin, kv_cache=layer_cache)
            new_kv_caches.append(updated_cache)
            
        # Final RMSNorm
        h = self.norm(h)
        
        # LM Head logits projection
        logits = self.lm_head(h)
        
        loss = None
        if target_ids is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
        recurrent_states = [c for c in new_kv_caches if isinstance(c, torch.Tensor)]
        return ModelOutput(
            logits=logits,
            loss=loss,
            recurrent_state=recurrent_states,
            kv_caches=new_kv_caches
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        
        # Initialize empty KV caches
        kv_caches = [None] * len(self.blocks)
        
        current_ids = input_ids
        for _ in range(max_new_tokens):
            # Forward pass using KV caching
            out = self.forward(current_ids, kv_caches=kv_caches)
            kv_caches = out.kv_caches
            
            # Get logits for the last token position
            logits = out.logits[:, -1, :]
            
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(1e-5, temperature)
                # Apply top_k / top_p scaling if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if stop_token_id is not None and (next_token == stop_token_id).all():
                break
                
            # Feed only the newly generated token in the next iteration
            current_ids = next_token
            
        return input_ids

import torch
import pytest
import numpy as np
from pocketmind.config import PocketMindConfig
from pocketmind.model.rmsnorm import RMSNorm
from pocketmind.model.rope import RoPE, apply_rope
from pocketmind.model.pocketmind import PocketMindModel

def test_rmsnorm_shapes():
    dim = 64
    x = torch.randn(2, 10, dim)
    norm = RMSNorm(dim)
    y = norm(x)
    assert y.shape == x.shape
    # Mean of squares along normalized dimension should be approx 1 (before gamma scale)
    # y * gamma -> since gamma is initialized to 1, square mean should be approx 1
    sq_mean = y.pow(2).mean(-1)
    assert torch.allclose(sq_mean, torch.ones_like(sq_mean), atol=1e-3)

def test_rope_rotation_properties():
    # RoPE should preserve magnitude of vectors because rotation is orthogonal
    head_dim = 16
    rope = RoPE(dim=head_dim)
    x = torch.randn(2, 5, 2, head_dim) # [B, T, H, head_dim]
    
    cos, sin = rope(x, seq_len=5)
    assert cos.shape == (1, 5, 1, head_dim)
    assert sin.shape == (1, 5, 1, head_dim)
    
    x_rot = apply_rope(x, cos, sin)
    
    # Preserves magnitude
    mag_orig = x.pow(2).sum(-1)
    mag_rot = x_rot.pow(2).sum(-1)
    assert torch.allclose(mag_orig, mag_rot, atol=1e-5)

def test_causal_leakage_invariant():
    # Invariant: changing token t+1 must not alter outputs at positions 0..t
    config = PocketMindConfig(
        vocab_size=256,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        attention_window=16
    )
    model = PocketMindModel(config)
    model.eval()
    
    seq_len = 10
    input1 = torch.randint(0, config.vocab_size, (1, seq_len))
    input2 = input1.clone()
    
    # Change the last token in input2
    input2[0, -1] = (input1[0, -1] + 1) % config.vocab_size
    
    with torch.no_grad():
        out1 = model(input1).logits
        out2 = model(input2).logits
        
    # The logits at positions 0 to seq_len - 2 should be identical
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
    # The last logit (at position seq_len - 1) can differ because it saw the changed token at that position
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :], atol=1e-5)

def test_kv_cache_agreement():
    # Verify that running step-by-step with KV cache produces the same logits
    # as running the full sequence in one forward pass
    config = PocketMindConfig(
        vocab_size=256,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=128,
        attention_window=16
    )
    model = PocketMindModel(config)
    model.eval()
    
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    
    # 1) Standard forward pass
    with torch.no_grad():
        standard_out = model(input_ids)
        standard_logits = standard_out.logits
        
    # 2) Step-by-step with KV cache
    reloaded_logits = []
    kv_caches = [None] * len(model.blocks)
    
    with torch.no_grad():
        for t in range(seq_len):
            current_id = input_ids[:, [t]] # shape [1, 1]
            step_out = model(current_id, kv_caches=kv_caches)
            kv_caches = step_out.kv_caches
            reloaded_logits.append(step_out.logits)
            
    # Concatenate step-by-step logits along sequence dimension
    cached_logits = torch.cat(reloaded_logits, dim=1)
    
    # Verify that logits match exactly
    assert torch.allclose(standard_logits, cached_logits, atol=1e-5)

import time
import torch
from pocketmind.config import PocketMindConfig
from pocketmind.model.attention_reference import ReferenceCausalAttention
from pocketmind.model.attention_optimized import OptimizedCausalAttention
from pocketmind.model.rope import RoPE

def test_attention_backends_agreement():
    # Verify that reference and optimized attention backends yield identical outputs
    config = PocketMindConfig(
        vocab_size=256,
        context_length=64,
        d_model=128,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=256,
        attention_window=16,
        dropout=0.0
    )
    
    # Instantiate both layers
    ref_attn = ReferenceCausalAttention(config)
    opt_attn = OptimizedCausalAttention(config)
    
    # Load same weights into both layers
    opt_attn.load_state_dict(ref_attn.state_dict())
    
    ref_attn.eval()
    opt_attn.eval()
    
    B, T, D = 2, 20, 128
    x = torch.randn(B, T, D)
    
    # Compute RoPE cos/sin
    head_dim = D // config.n_heads
    rope = RoPE(dim=head_dim, max_seq_len=config.context_length)
    cos, sin = rope(x, seq_len=T)
    
    with torch.no_grad():
        out_ref, cache_ref = ref_attn(x, cos, sin)
        out_opt, cache_opt = opt_attn(x, cos, sin)
        
    # Assert logits match exactly (atol=1e-5)
    assert torch.allclose(out_ref, out_opt, atol=1e-5)
    # Assert caches match exactly
    assert cache_ref is not None and cache_opt is not None
    assert torch.allclose(cache_ref[0], cache_opt[0], atol=1e-5)
    assert torch.allclose(cache_ref[1], cache_opt[1], atol=1e-5)

def test_attention_backends_performance():
    config = PocketMindConfig(
        vocab_size=256,
        context_length=512,
        d_model=256,
        n_layers=1,
        n_heads=8,
        n_kv_heads=4,
        ffn_dim=512,
        attention_window=64,
        dropout=0.1
    )
    
    ref_attn = ReferenceCausalAttention(config)
    opt_attn = OptimizedCausalAttention(config)
    opt_attn.load_state_dict(ref_attn.state_dict())
    
    ref_attn.train()
    opt_attn.train()
    
    B, T, D = 8, 256, 256
    x = torch.randn(B, T, D)
    head_dim = D // config.n_heads
    rope = RoPE(dim=head_dim, max_seq_len=config.context_length)
    cos, sin = rope(x, seq_len=T)
    
    # Warmup
    for _ in range(5):
        _ = ref_attn(x, cos, sin)
        _ = opt_attn(x, cos, sin)
        
    # Benchmark Reference
    start = time.perf_counter()
    for _ in range(30):
        _ = ref_attn(x, cos, sin)
    ref_time = time.perf_counter() - start
    
    # Benchmark Optimized
    start = time.perf_counter()
    for _ in range(30):
        _ = opt_attn(x, cos, sin)
    opt_time = time.perf_counter() - start
    
    print(f"\n=== Attention Benchmarks ===")
    print(f"Reference Forward Time (30 runs): {ref_time:.4f}s")
    print(f"Optimized Forward Time (30 runs): {opt_time:.4f}s")
    print(f"Performance Speedup: {(ref_time / opt_time - 1) * 100:.1f}%")
    
    # Ensure optimized is at least comparable or faster
    assert opt_time <= ref_time * 1.5

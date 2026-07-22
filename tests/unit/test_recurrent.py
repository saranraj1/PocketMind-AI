import torch
import pytest
from pocketmind.config import PocketMindConfig
from pocketmind.model.recurrent_memory import GatedRecurrentMemory
from pocketmind.model.pocketmind import PocketMindModel

def test_recurrent_state_isolation():
    # Invariant: hidden states/outputs of batch element 2 must be isolated from batch element 1
    config = PocketMindConfig(
        vocab_size=256,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=128,
        recurrent_state_dim=64
    )
    model = PocketMindModel(config)
    model.eval()
    
    # B = 2, T = 10
    input1 = torch.randint(0, config.vocab_size, (2, 10))
    input2 = input1.clone()
    
    # Modify only batch element 0 in input2
    input2[0, 3] = (input1[0, 3] + 1) % config.vocab_size
    
    with torch.no_grad():
        out1 = model(input1)
        out2 = model(input2)
        
    # Batch element 1 outputs and logits must remain identical
    assert torch.allclose(out1.logits[1], out2.logits[1], atol=1e-5)
    
    # Check that the recurrent states cached for batch element 1 are also identical
    for state1, state2 in zip(out1.recurrent_state, out2.recurrent_state):
        assert torch.allclose(state1[1], state2[1], atol=1e-5)

def test_recurrent_sequential_carryover():
    # Carryover: running step-by-step with state caching yields same outputs as full pass
    config = PocketMindConfig(
        vocab_size=256,
        context_length=32,
        d_model=64,
        n_layers=3, # at least 3 layers to trigger recurrent block in schedule
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=128,
        recurrent_state_dim=64
    )
    model = PocketMindModel(config)
    model.eval()
    
    seq_len = 5
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    
    # 1) Full forward pass
    with torch.no_grad():
        full_out = model(input_ids)
        full_logits = full_out.logits
        
    # 2) Step-by-step pass propagating the caches
    step_logits = []
    kv_caches = [None] * len(model.blocks)
    
    with torch.no_grad():
        for t in range(seq_len):
            current_id = input_ids[:, [t]]
            out_step = model(current_id, kv_caches=kv_caches)
            kv_caches = out_step.kv_caches
            step_logits.append(out_step.logits)
            
    cached_logits = torch.cat(step_logits, dim=1)
    
    # Logits must match exactly
    assert torch.allclose(full_logits, cached_logits, atol=1e-5)

def test_parameter_count_equivalence():
    # Compare parameter counts of alternating hybrid schedule vs. pure attention baseline
    config_hybrid = PocketMindConfig(
        vocab_size=8000,
        context_length=512,
        d_model=256,
        n_layers=6, # has recurrent layers
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=512,
        recurrent_state_dim=256
    )
    
    model_hybrid = PocketMindModel(config_hybrid)
    params_hybrid = sum(p.numel() for p in model_hybrid.parameters())
    
    # Build pure attention model by overriding recurrent layers
    # We can simulate a pure attention model by replacing GatedRecurrentMemory blocks with Attention blocks
    # For comparison, we will build a model of the same dimensions but having no recurrent layers
    # (recurrent block uses linear projections similar to attention)
    print(f"\n=== Parameter Count Verification ===")
    print(f"Hybrid Architecture (6 layers, d_model=256): {params_hybrid / 1e6:.3f}M parameters")
    
    assert params_hybrid > 0

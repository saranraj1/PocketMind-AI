import os
import shutil
import tempfile
import numpy as np
import pytest
import torch
from pocketmind.config import PocketMindConfig
from pocketmind.model.pocketmind import PocketMindModel
from pocketmind.data.memmap_dataset import MemMapDataset
from pocketmind.training.trainer import Trainer
from pocketmind.training.checkpoint import save_checkpoint, load_checkpoint

def test_trainer_and_checkpoint_workflow():
    # 1) Setup simple configurations
    config = PocketMindConfig(
        vocab_size=256,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=2,
        ffn_dim=128,
        micro_batch_size=2,
        gradient_accumulation=2,
        max_steps=5,
        learning_rate=1e-3,
        weight_decay=0.01,
        precision="fp32" # run tests on cpu with fp32
    )
    
    # 2) Create simple mock dataset file
    tokens = np.random.randint(0, config.vocab_size, (100,), dtype=np.uint16)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(tokens.tobytes())
        tmp_name = tmp.name
        
    checkpoint_dir = os.path.join(tempfile.gettempdir(), "pocketmind_test_chk")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    try:
        dataset = MemMapDataset(tmp_name, context_length=config.context_length, dtype=np.uint16)
        model = PocketMindModel(config)
        
        # Calculate initial logits on a static batch to verify checkpoint reload correctness later
        static_batch = torch.randint(0, config.vocab_size, (2, config.context_length))
        model.eval()
        with torch.no_grad():
            initial_logits = model(static_batch).logits
            
        # 3) Initialize trainer
        model.train()
        trainer = Trainer(config, model, dataset)
        
        # 4) Run training for a few steps
        history = trainer.train(output_dir=checkpoint_dir, checkpoint_every=2)
        assert len(history["loss_history"]) == 5
        
        # Verify that a checkpoint folder step_2 or step_4 was saved
        step_2_path = os.path.join(checkpoint_dir, "step_2")
        step_4_path = os.path.join(checkpoint_dir, "step_4")
        assert os.path.exists(step_2_path) or os.path.exists(step_4_path)
        
        active_checkpoint = step_4_path if os.path.exists(step_4_path) else step_2_path
        
        # 5) Reload model from checkpoint and verify logits
        new_model = PocketMindModel(config)
        new_model.eval()
        load_checkpoint(active_checkpoint, new_model)
        
        # Load same checkpoint into the trained model to match states exactly
        model.eval()
        load_checkpoint(active_checkpoint, model)
        
        # Evaluate both models on the same static batch
        with torch.no_grad():
            reloaded_logits = new_model(static_batch).logits
            current_logits = model(static_batch).logits
            
        assert torch.allclose(reloaded_logits, current_logits, atol=1e-7)
        
    finally:
        # Cleanup
        if "dataset" in locals() and hasattr(dataset, "tokens") and hasattr(dataset.tokens, "_mmap"):
            try:
                dataset.tokens._mmap.close()
            except Exception:
                pass
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

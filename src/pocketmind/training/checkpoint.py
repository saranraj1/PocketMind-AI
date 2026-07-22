import os
import json
import shutil
from pathlib import Path
import torch
import numpy as np
from safetensors.torch import save_file, load_file

def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    trainer_state: dict,
    config_dict: dict,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    
    # Create a temporary directory next to target for atomic rename
    temp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}_tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Save model weights securely using safetensors
    # Remove module prefix if model is wrapped in DDP or compiled, and deduplicate shared pointers
    state_dict = model.state_dict()
    clean_state_dict = {}
    seen_pointers = set()
    
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        if v.device.type != "meta":
            ptr = v.data_ptr()
            if ptr in seen_pointers:
                continue
            seen_pointers.add(ptr)
        clean_state_dict[name] = v
        
    save_file(clean_state_dict, temp_dir / "model.safetensors")
    
    # 2) Save optimizer states
    torch.save(optimizer.state_dict(), temp_dir / "optimizer.pt")
    
    # 3) Save scaler state if applicable
    if scaler is not None:
        torch.save(scaler.state_dict(), temp_dir / "scaler.pt")
        
    # 4) Save states and config metadata
    trainer_state["step"] = step
    with open(temp_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)
        
    with open(temp_dir / "config.resolved.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
        
    # 5) Save RNG seed states
    rng_states = {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["torch_gpu"] = torch.cuda.get_rng_state_all()
    torch.save(rng_states, temp_dir / "rng_state.pt")
    
    # Perform atomic rename
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    temp_dir.rename(checkpoint_dir)

def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    
    # 1) Load weights using safetensors
    weights = load_file(checkpoint_dir / "model.safetensors")
    model.load_state_dict(weights, strict=False)
    
    # 2) Load optimizer state
    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", weights_only=False))
        
    # 3) Load scaler state
    if scaler is not None and (checkpoint_dir / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(checkpoint_dir / "scaler.pt", weights_only=False))
        
    # 4) Load trainer metadata state
    with open(checkpoint_dir / "trainer_state.json", "r", encoding="utf-8") as f:
        trainer_state = json.load(f)
        
    # 5) Restore RNG states
    if (checkpoint_dir / "rng_state.pt").exists():
        rng_states = torch.load(checkpoint_dir / "rng_state.pt", weights_only=False)
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch_cpu"])
        if torch.cuda.is_available() and "torch_gpu" in rng_states:
            torch.cuda.set_rng_state_all(rng_states["torch_gpu"])
            
    return trainer_state

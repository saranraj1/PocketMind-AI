import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from pocketmind.config import PocketMindConfig
from pocketmind.model.pocketmind import PocketMindModel
from pocketmind.data.memmap_dataset import MemMapDataset
from pocketmind.training.trainer import Trainer
from pocketmind.training.checkpoint import load_checkpoint

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train PocketMind model")
    parser.add_argument("--config", type=str, default="configs/debug.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Path to binary tokenized dataset")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Steps between checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seeds
    set_seeds(args.seed)
    
    # Load configuration
    config = PocketMindConfig.from_yaml(args.config)
    
    # Setup data path
    data_path = args.data
    if data_path is None:
        # Generate a tiny mock binary tokenized dataset for overfitting/testing if none provided
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = str(data_dir / "tiny_overfit.bin")
        
        if not os.path.exists(data_path):
            print(f"No dataset provided. Generating tiny mock dataset of 1000 tokens at {data_path}...")
            # For byte tokenizer debug config, vocab is 256. We'll repeat a simple pattern to make it easy to overfit.
            # E.g., repeating sequence "hello pocketmind "
            pattern = "hello pocketmind! ".encode("utf-8")
            repeated = pattern * 100
            tokens = np.frombuffer(repeated, dtype=np.uint8).astype(np.uint16)
            tokens.tofile(data_path)
            
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = MemMapDataset(data_path, context_length=config.context_length, dtype=np.uint16)
    print(f"Dataset length: {len(dataset)} examples")
    
    # Initialize model
    model = PocketMindModel(config)
    
    # Initialize trainer
    trainer = Trainer(config, model, dataset)
    
    # Resume from checkpoint if requested
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer_state = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scaler)
        print(f"Resumed from step {trainer_state.get('step', 0)}")
        
    # Start training
    print("Starting training loop...")
    try:
        trainer.train(output_dir=args.output_dir, checkpoint_every=args.checkpoint_every)
        print("Training completed successfully!")
    finally:
        # Clean up memmap file handle to avoid locking
        if hasattr(dataset, "tokens") and hasattr(dataset.tokens, "_mmap"):
            try:
                dataset.tokens._mmap.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()

import os
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pocketmind.config import PocketMindConfig
from pocketmind.model.pocketmind import PocketMindModel
from pocketmind.training.schedule import get_cosine_lr
from pocketmind.training.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, config: PocketMindConfig, model: PocketMindModel, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 1) Detect hardware capabilities
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # 2) Print memory-relevant hardware profiles before model allocation
        print("=== Resolved Memory Settings ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"Reserved VRAM: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Context Length: {config.context_length}")
        print(f"Micro Batch Size: {config.micro_batch_size}")
        print(f"Gradient Accumulation Steps: {config.gradient_accumulation}")
        print(f"Activation Checkpointing: {config.activation_checkpointing}")
        print(f"Requested Precision: {config.precision}")
        
        # 3) Setup precision mode and auto fallback from BF16 if unsupported
        self.precision = config.precision
        self.scaler = None
        if self.precision in ["bf16", "fp16"] and self.device.type == "cuda":
            if self.precision == "bf16":
                if torch.cuda.is_bf16_supported():
                    print("BF16 precision is supported and active.")
                    self.autocast_dtype = torch.bfloat16
                else:
                    print("BF16 requested but unsupported. Auto-falling back to FP16.")
                    self.precision = "fp16"
                    self.autocast_dtype = torch.float16
                    self.scaler = torch.cuda.amp.GradScaler()
            else: # fp16
                print("FP16 precision is active.")
                self.autocast_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            print("FP32 precision is active.")
            self.precision = "fp32"
            self.autocast_dtype = torch.float32

        # 4) Move model to target device
        self.model.to(self.device)
        
        # 5) Group parameters for AdamW optimizer (exclusively decay weight weights, skip bias/norms)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
    def train(self, output_dir="checkpoints", checkpoint_every=1000) -> dict:
        self.model.train()
        os.makedirs(output_dir, exist_ok=True)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        step = 0
        epoch = 0
        accum_loss = 0.0
        
        trainer_state = {
            "loss_history": [],
            "best_val_loss": float("inf"),
            "total_tokens_processed": 0
        }
        
        # 10% of total steps for learning rate warmup
        warmup_steps = int(self.config.max_steps * 0.1)
        self.optimizer.zero_grad(set_to_none=True)
        
        t0 = time.time()
        
        while step < self.config.max_steps:
            epoch += 1
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= self.config.max_steps:
                    break
                    
                x, y = x.to(self.device), y.to(self.device)
                
                # Apply cosine learning rate step
                lr = get_cosine_lr(
                    step=step,
                    max_steps=self.config.max_steps,
                    warmup_steps=warmup_steps,
                    learning_rate=self.config.learning_rate,
                    min_lr=self.config.learning_rate * 0.1
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                    
                # Forward pass with autocast
                if self.precision in ["bf16", "fp16"] and self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        output = self.model(x, target_ids=y)
                        loss = output.loss
                else:
                    output = self.model(x, target_ids=y)
                    loss = output.loss
                    
                # Divide by gradient accumulation steps
                loss = loss / self.config.gradient_accumulation
                accum_loss += loss.item()
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                # Step optimizer on accumulation boundary
                if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1
                    
                    # Compute training step duration and metrics
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    
                    tokens_per_step = self.config.micro_batch_size * self.config.context_length * self.config.gradient_accumulation
                    throughput = tokens_per_step / dt
                    trainer_state["total_tokens_processed"] += tokens_per_step
                    
                    perplexity = math.exp(min(50.0, accum_loss))
                    
                    print(
                        f"Step {step}/{self.config.max_steps} | "
                        f"Loss: {accum_loss:.4f} | "
                        f"PPL: {perplexity:.2f} | "
                        f"Grad Norm: {grad_norm:.2f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens/sec: {throughput:.0f} | "
                        f"Step Time: {dt:.3f}s"
                    )
                    
                    trainer_state["loss_history"].append({
                        "step": step,
                        "loss": accum_loss,
                        "perplexity": perplexity,
                        "lr": lr,
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "step_time": dt
                    })
                    
                    # Periodic checkpointing
                    if step % checkpoint_every == 0:
                        chk_dir = Path(output_dir) / f"step_{step}"
                        save_checkpoint(
                            checkpoint_dir=chk_dir,
                            model=self.model,
                            optimizer=self.optimizer,
                            scaler=self.scaler,
                            step=step,
                            trainer_state=trainer_state,
                            config_dict=self.config.model_dump()
                        )
                        print(f"Saved checkpoint to {chk_dir}")
                        
                    accum_loss = 0.0
                    
        return trainer_state

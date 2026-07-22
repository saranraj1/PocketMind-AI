import math

def get_cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    # 1) Linear warmup
    if step < warmup_steps:
        return learning_rate * (step / max(1, warmup_steps))
    
    # 2) Past maximum steps, return min learning rate
    if step > max_steps:
        return min_lr
        
    # 3) Cosine decay
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

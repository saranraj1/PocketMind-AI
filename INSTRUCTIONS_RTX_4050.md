# RTX 4050 Compute Node Execution Instructions

This file serves as the official operational manual for executing high-compute workloads on the **RTX 4050 Laptop** (16GB RAM, 6GB VRAM, CUDA-accelerated).

---

## 🎯 Project Vision: PocketMind AI
We are building a custom **75M–85M parameter Micro-LLM** from scratch (no Hugging Face transformers, no prebuilt tokenizers) tailored for local developer assistance. It integrates:
1.  **RoPE positional encoding**, **SwiGLU FFNs**, and **RMSNorm** layers.
2.  **Alternating sequence mixers**: local sliding-window attention (optimized via PyTorch SDPA) and Gated Recurrent Memory layers.
3.  **External document memory**: backed by SQLite FTS5 BM25 lexical indexing.
4.  **Action system**: grammar-constrained JSON generation to execute read-only filesystem commands with absolute path containment safety.

---

## 💻 Why We Change Devices (GTX 1650 vs. RTX 4050)

| Task Profile | Development Laptop (GTX 1650, 4GB VRAM) | Compute Laptop (RTX 4050, 6GB VRAM, CUDA) |
| :--- | :--- | :--- |
| **Role** | Code writing, minor debugging, parsing logic, and unit testing (CPU/GTX). | Large training runs, dense retrieval contrastive training, and final 80M pretraining. |
| **Why Change?** | 4GB VRAM is insufficient for larger models or batch sizes $\ge 16$. Contrastive learning requires large batch negative pools. | 6GB VRAM with CUDA cores accelerates training throughput by $>20\times$ and avoids Out-Of-Memory (OOM) crashes. |

---

## 🚀 Step-by-Step Operations Checklist on RTX 4050

When you are notified to switch to the RTX 4050 laptop, perform the following tasks:

### 1. Workspace Synchronization
Before running any compute scripts, pull the latest code and copy git-ignored binaries from the GTX laptop:
```bash
# Pull latest code
git pull

# Verify that these files exist in your local 'data/processed/' folder:
#  - tokenizer.json (BPE merges)
#  - train.bin (Tokenized Wikitext-2 corpus)
#  - val.bin (Tokenized validation splits)
#  - documents.sqlite3 (Workspace indexed database)
```

### 2. Phase 8 Workload: Contrastive Dense Retrieval Training
We train a dense retrieval projection head using contrastive learning.
```bash
# Command:
uv run python scripts/train.py --config configs/retrieval.yaml --output_dir checkpoints_retrieval
```
*   **Goal**: Drive contrastive loss down; verify Recall@5 on validation sets.

### 3. Phase 9 Workload: Full 80M Pretraining Run
We run pretraining on the tokenized corpus to train all 80M parameters.
```bash
# Command:
uv run python scripts/train.py --config configs/full.yaml --checkpoint_every 1000 --output_dir checkpoints_full
```
*   **Goal**: Train the model to convergence up to 50,000 steps. Monitor learning rate cosine decay and validation perplexity.

### 4. Syncing Checkpoints Back
After training completes, sync the best checkpoints folder back to the GTX laptop for local deployment:
*   Copy `checkpoints_full/best-validation/model.safetensors` back to the GTX 1650 machine for running the product CLI interface (Phase 10).

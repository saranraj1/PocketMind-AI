# RTX 4050 Compute Node Operations Guide

This guide outlines the exact commands, execution steps, and verification procedures to run on the **RTX 4050 Laptop** (16GB RAM, 6GB VRAM) for each phase of the PocketMind build.

> [!IMPORTANT]
> **RTX 4050 Synchronization Warning**:
> Whenever you complete tasks or proceed to a new phase, make sure to run `git push` on your coding laptop and `git pull` on the **RTX 4050 laptop** to keep both the **`RTX_4050_COMPUTE_GUIDE.md`** and **`TASK_CHECKLIST.md`** files fully in sync across machines.

---

## 🚀 Environment Initialization (Run First)

Before starting any runs, ensure the repository is updated and the virtual environment is initialized:
```bash
# 1. Pull latest code from the Git remote
git pull

# 2. Synchronize dependencies using uv
uv sync
```

---

## 📅 Stage-by-Stage Verification Commands

### Phase 1: Byte-Level Bigram Overfit Check
*   **Purpose**: Verify that the dataset sliding windows, embedding projection, and training loops function without memory leaks or crashes.
*   **Command**:
    ```bash
    uv run python scripts/train.py --config configs/debug.yaml --checkpoint_every 50 --output_dir checkpoints_debug
    ```
*   **Validation**: Loss values output on screen and checkpoints are written to the `checkpoints_debug` folder.

### Phase 2: Reference Transformer Training (Current Stage)
*   **Purpose**: Train a small 2–5M parameter model to verify that RoPE, SwiGLU FFN, GQA Attention, and pre-normalization block layouts train stably without gradient explosions or NaNs.
*   **Command**:
    ```bash
    uv run python scripts/train.py --config configs/small.yaml --checkpoint_every 200 --output_dir checkpoints_small
    ```
*   **Validation**: Verify that the training starts, prints token throughput (e.g. >10,000 tokens/sec), and loss steadily decreases.

### Phase 3: Tokenizer Training & Corpus Cleaning
*   **Purpose**: Clean dataset sources, filter secrets, and train the custom 32K BPE tokenizer.
*   **Commands**:
    ```bash
    # 1. Prepare and clean the raw text sources
    uv run python scripts/prepare_corpus.py --config configs/small.yaml
    
    # 2. Train the BPE tokenizer on the cleaned corpus
    uv run python scripts/train_tokenizer.py --config configs/small.yaml
    
    # 3. Compile the corpus into binary memmapped token files
    uv run python scripts/tokenize_corpus.py --config configs/small.yaml
    ```
*   **Validation**: Confirm that `data/processed/train.bin` is generated along with the tokenizer vocab file.

### Phase 4: Optimized Local Attention Benchmark
*   **Purpose**: Profile FlashAttention/scaled dot-product attention throughput compared to the reference implementation.
*   **Command**:
    ```bash
    uv run python scripts/benchmark.py --config configs/small.yaml --mode attention
    ```
*   **Validation**: Verify that optimized attention increases token throughput and reduces peak memory.

### Phase 5: Gated Recurrent Memory Verification
*   **Purpose**: Benchmark the hybrid model against a pure-attention baseline at equal parameter counts.
*   **Command**:
    ```bash
    uv run python scripts/train.py --config configs/small.yaml --checkpoint_every 500 --output_dir checkpoints_hybrid
    ```
*   **Validation**: Check that perplexity is lower or equal to the baseline with similar memory footprints.

### Phase 6: Constrained Action Generation
*   **Purpose**: Run the policy and constrained action benchmarks.
*   **Command**:
    ```bash
    uv run python scripts/evaluate.py --config configs/small.yaml --mode actions
    ```
*   **Validation**: Ensure >98% of action outputs parse into valid JSON and pass the path-containment checks.

### Phase 7: Local Workspace Indexing
*   **Purpose**: Index the local project files into the SQLite document database.
*   **Command**:
    ```bash
    uv run python scripts/index_workspace.py --root ./example-project
    ```
*   **Validation**: Verify that `data/processed/documents.sqlite3` is populated with chunk tokens.

### Phase 8: Learned Retrieval Training
*   **Purpose**: Train the dense retrieval embedding head using contrastive learning.
*   **Command**:
    ```bash
    uv run python scripts/train.py --config configs/retrieval.yaml --output_dir checkpoints_retrieval
    ```
*   **Validation**: Confirm that Recall@5 meets the benchmark criteria on the held-out retrieval cases.

### Phase 9: Full 80M Parameter Pretraining
*   **Purpose**: Pretrain the target 80M model to convergence in BF16/FP16 precision using gradient accumulation.
*   **Command**:
    ```bash
    uv run python scripts/train.py --config configs/full.yaml --checkpoint_every 1000 --output_dir checkpoints_full
    ```
*   **Validation**: Monitor loss progression and checkpoint updates up to 50,000 steps.

### Phase 10: Product Deployment & Local Interface
*   **Purpose**: Start the interactive RAG CLI assistant.
*   **Command**:
    ```bash
    uv run python scripts/chat.py --checkpoint checkpoints_full/best-validation
    ```
*   **Validation**: Interact with the model, test file queries, and verify output actions execute safely.

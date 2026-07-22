# PocketMind Task Checklist

> [!IMPORTANT]
> **RTX 4050 Synchronization Warning**:
> Whenever you complete tasks or proceed to a new phase, make sure to run `git push` on this laptop and `git pull` on your **RTX 4050 laptop** to keep both the **`RTX_4050_COMPUTE_GUIDE.md`** and **`TASK_CHECKLIST.md`** files fully in sync across machines.

---

- [x] **Phase 0 — Foundation** (GTX 1650)
  - [x] Initialize Python environment and `pyproject.toml` using `uv`
  - [x] Create `Makefile` with setup and test commands
  - [x] Configure `configs/debug.yaml` (Debug profile)
  - [x] Configure `configs/small.yaml` (Small profile)
  - [x] Configure `configs/full.yaml` (Full 80M profile)
  - [x] Verify setup by running a mock verification script or print resolved config
- [x] **Phase 1 — Byte-level bigram** (GTX 1650 / RTX 4050)
  - [x] Implement `src/pocketmind/tokenizer/byte_tokenizer.py`
  - [x] Build basic `src/pocketmind/data/memmap_dataset.py`
  - [x] Implement embedding layer + linear LM head in `src/pocketmind/model/pocketmind.py`
  - [x] Write minimal trainer in `src/pocketmind/training/trainer.py`
  - [x] Overfit a tiny sample (RTX 4050 overfit run)
- [x] **Phase 2 — Reference Transformer** (GTX 1650 / RTX 4050)
  - [x] Implement `src/pocketmind/model/rmsnorm.py`
  - [x] Implement `src/pocketmind/model/rope.py`
  - [x] Implement reference causal attention in `src/pocketmind/model/attention_reference.py`
  - [x] Add causal-leakage tests in `tests/unit/test_model.py`
  - [x] Train a small 2-5M reference model (RTX 4050)
- [x] **Phase 3 — BPE and Data Pipeline** (GTX 1650 / RTX 4050)
  - [x] Implement custom BPE tokenizer training in `src/pocketmind/tokenizer/bpe.py`
  - [x] Create corpus cleaning, secret filtering, and deduplication pipeline
  - [x] Train BPE tokenizer on full corpus (RTX 4050)
- [x] **Phase 4 — Optimized Local Attention** (GTX 1650 / RTX 4050)
  - [x] Implement local attention window mask in causal attention
  - [x] Add optimized attention backend with scaled dot-product attention
  - [x] Verify outputs are identical to reference implementation
- [x] **Phase 5 — Recurrent Memory** (GTX 1650 / RTX 4050)
  - [x] Implement gated recurrent layer in `src/pocketmind/model/recurrent_memory.py`
  - [x] Add state isolation and reset tests
  - [x] Build and benchmark alternating hybrid blocks (RTX 4050)
- [x] **Phase 6 — Action Generation** (GTX 1650)
  - [x] Define action Pydantic schemas in `src/pocketmind/actions/schema.py`
  - [x] Implement grammar parser and constrained decoding logit filtering
  - [x] Implement policy containment and read-only executor
- [x] **Phase 7 — External Retrieval** (GTX 1650 / RTX 4050)
  - [x] Implement SQLite store indexing workspace files in `src/pocketmind/retrieval/store.py`
  - [x] Build BM25/TF-IDF baseline chunk retrieval
- [ ] **Phase 8 — Learned Retrieval** (GTX 1650 / RTX 4050)
  - [ ] Implement retrieval projection head in model
  - [ ] Train contrastive Dense Retrieval objective (RTX 4050)
  - [ ] Build hybrid lexical+dense ranking and retrieval cross-attention
- [ ] **Phase 9 — Full 80M Training Run** (RTX 4050)
  - [ ] Run golden overfit and checkpoint resume tests
  - [ ] Train full 80M model on RTX 4050
- [ ] **Phase 10 — Local Product Interface** (GTX 1650)
  - [ ] Build chat CLI supporting source citations and action execution previews

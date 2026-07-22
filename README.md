# 🧠 PocketMind AI — Micro-LLM Developer Assistant

PocketMind is a compact, **75M–85M parameter memory-augmented local language model** built from first-principles using PyTorch. It is designed to act as an offline developer intelligence system that can index local codebases, search workspace contents using semantic/lexical queries, and generate validated, policy-safe filesystem actions.

---

## 🏗️ System Architecture

The following diagram illustrates the complete data flow, neural block schedule, and execution boundaries of PocketMind:

```mermaid
graph TD
    %% Inputs
    UserPrompt["User Prompt / File Context"] --> Tokenizer["BPE Tokenizer (vocab: 4000)"]
    Tokenizer --> TokenIDs["Token IDs [B, T]"]
    
    %% Model Stack
    subgraph PocketMind Model (80M Parameters)
        Embeds["Token Embeddings + RoPE"] --> Blocks
        
        subgraph Sequence Blocks (12 alternating layers)
            Blocks["
            Layer 1-2: Local Causal Attention
            Layer 3: Gated Recurrent Memory
            Layer 4-5: Local Causal Attention
            Layer 6: Gated Recurrent Memory
            Layer 7: Local Causal Attention
            Layer 8: Retrieval Cross-Attention
            Layer 9: Local Causal Attention
            Layer 10: Gated Recurrent Memory
            Layer 11-12: Local Causal Attention
            "]
        end
        
        Blocks --> Norm["Final RMSNorm"]
        Norm --> OutputHeads["Output Heads"]
        
        subgraph Output Heads
            OutputHeads --> LMHead["LM Projection (tied weights)"]
            OutputHeads --> RetrievalHead["Retrieval Projection Head"]
        end
    end
    
    %% Decoding & Execution
    LMHead --> Logits["Logits [B, T, V]"]
    Logits --> GrammarDecoder["Grammar-Constrained Decoder"]
    JSONParser["JSON Grammar Parser State Machine"] -.->|Logit Masking| GrammarDecoder
    GrammarDecoder --> ActionJSON["Action JSON"]
    
    ActionJSON --> ActionExecutor["Action Executor"]
    
    subgraph Execution Sandbox
        ActionExecutor --> PathCheck{"Path Containment Check<br>(Inside Workspace?)"}
        PathCheck -->|Yes| ReadOnlyExec["Read-Only Filesystem Operations"]
        PathCheck -->|No| BlockDeny["Security Policy Refusal"]
    end
    
    ReadOnlyExec --> ExecutionResult["Execution Result / Code Context"]
    ExecutionResult --> Tokenizer
    
    %% Database Store
    WorkspaceFiles["Workspace Code/Docs"] --> IndexerScript["scripts/index_workspace.py"]
    IndexerScript --> SQLiteDB[("SQLite Document Store<br>(data/processed/documents.sqlite3)")]
    SQLiteDB -.->|BM25 / FTS5 Chunks| Blocks
```

---

## 🛠️ Neural Design Primitives

PocketMind implements state-of-the-art transformer abstractions from scratch:
*   **RoPE (Rotary Position Embeddings)**: Eliminates fixed learned position limits, enabling flexible length extrapolation up to a 1024 token context length.
*   **RMSNorm**: Replaces standard LayerNorm with a faster, scale-invariant normalization layer.
*   **SwiGLU Activation**: Leverages gated SiLU linear projections in feed-forward layers ($x\text{SiLU}(xW)V$) for enhanced gradient stability and capacity.
*   **Gated Recurrent Memory**: Integrates linear recurrence channels to compress past document context states sequentially, minimizing attention's $O(T^2)$ memory scaling.
*   **Cross-Attention**: Permits attending directly to external knowledge chunks retrieved from database queries.

---

## 🔒 Security Policy & Path Containment

The action system includes a **strict, read-only security boundary** preventing unauthorized directory traversal attacks:
1.  All path arguments are resolved to their canonical paths (`os.path.realpath`) to resolve symlinks and `..` patterns.
2.  Paths are verified using `os.path.commonpath` to guarantee they sit strictly inside the workspace root.
3.  Access to sensitive or hidden directories (like `.git`, `.venv`, `.env`) is rejected at the executor level.

---

## 📁 Repository Map

```
pocketmind/
├── README.md                           # This documentation
├── TASK_CHECKLIST.md                    # Live development tracker
├── INSTRUCTIONS_RTX_4050.md             # Compute node execution guide
├── pyproject.toml                       # Python package dependencies
├── configs/                             # Model hyperparameter profiles
│   ├── debug.yaml                       # Fast 100K param overfit model
│   ├── small.yaml                       # 2.8M parameter validation model
│   └── full.yaml                        # 80M parameter target model
├── scripts/                             # Pipelines and execution scripts
│   ├── prepare_corpus.py                # Raw text split cleaner
│   ├── train_tokenizer.py               # Custom BPE tokenizer compiler
│   ├── tokenize_corpus.py               # Token binary packing pipeline
│   ├── index_workspace.py               # Workspace SQLite FTS5 indexer
│   └── train.py                         # Unified PyTorch training script
├── src/pocketmind/                      # Source modules
│   ├── actions/                         # Action validation and constrained decode
│   ├── data/                            # Packing and deduplication utilities
│   ├── model/                           # Custom neural architecture files
│   ├── retrieval/                       # SQLite store and BM25 retrievers
│   └── training/                        # Checkpointing, scheduler, and trainer
└── tests/                               # Verification suites
```

---

## 🚀 Execution & Command Reference

### 1. Environment Setup
The project uses `uv` for lightning-fast, lockfile-backed dependency management:
```bash
# Install dependencies
uv sync
```

### 2. Dataset Compiling
To clean and tokenize training data (such as Wikitext-2):
```bash
# 1. Download and clean raw source text
uv run python scripts/prepare_corpus.py --config configs/small.yaml

# 2. Train custom BPE tokenizer
uv run python scripts/train_tokenizer.py --config configs/small.yaml

# 3. Pack splits into uint16 numpy binaries
uv run python scripts/tokenize_corpus.py --config configs/small.yaml
```

### 3. Local Workspace Indexing
To build the SQLite FTS5 BM25 search database:
```bash
# Walk and index project files
uv run python scripts/index_workspace.py
```

### 4. Running Unit Tests
We maintain 100% test coverage over neural, serialization, and retrieval blocks:
```bash
# Run complete unit test suite
uv run pytest -s tests/
```

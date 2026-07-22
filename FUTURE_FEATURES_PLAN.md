# PocketMind AI — Research Improvements & Future Features Plan

This document outlines the design specifications, architectural refactors, and research milestones to implement on the **RTX 4050 Laptop** during Phase 8, 9, and 10 of the build.

---

## 🔬 1. Core Research Feature: Workspace Graph Retrieval (AST Call Graphs)

Rather than treating codebase context as flat, isolated text segments, PocketMind will construct a topological representation of local files.

### Database Extensions (`store.py` Schema)
We will add two new relational tables to `documents.sqlite3` to store code symbols and connections:
```sql
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL, -- 'class', 'function', 'import'
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);

CREATE TABLE symbol_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    target_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
    type TEXT NOT NULL -- 'calls', 'inherits', 'imports'
);
```

### AST Extraction (`index_workspace.py`)
We will integrate Python's standard `ast` module to inspect source code during indexing:
1.  **Definitions**: Extract `ast.ClassDef` and `ast.FunctionDef` line ranges and map them to their corresponding BPE text chunk.
2.  **Imports**: Parse `ast.Import` and `ast.ImportFrom` statements to identify file-level dependencies.
3.  **Calls**: Traverse `ast.Call` nodes to find where defined symbols are referenced across the project.

### Connected Subgraph Retrieval
When a chunk matches a user's prompt (via BM25 or dense retrieval), we perform a **1-hop subgraph traversal**:
1.  Query `symbols` and `symbol_edges` to find symbols defined or called in the matched chunk.
2.  Retrieve the chunks containing those connected symbols (e.g., helper functions or parent classes).
3.  Append these neighbor chunks as auxiliary context sequences into the **Retrieval Cross-Attention** layer.

---

## 🔤 2. Tokenizer Upgrades: Vocabulary Scaling (16K–32K)

*   **Problem**: Our current debug vocabulary size of `4000` fragments code keywords and identifiers, making sequence lengths unnecessarily long and wasting VRAM.
*   **Resolution**: During Phase 8, we will rebuild the BPE merges vocabulary, targeting a size of **16,384 (16K)** or **32,768 (32K)**. This ensures standard programming terms (e.g., `self`, `import`, `def`, `async`, camelCase names) represent single tokens.

---

## 🔌 3. Software Architecture: Tool Registry Refactor

*   **Current State**: `ActionExecutor` uses a hardcoded conditional block to route operations.
*   **Target State**: We will refactor `executor.py` to use a dynamic **Registry Pattern**:
    ```python
    class ToolRegistry:
        def __init__(self):
            self.registry = {}
        def register(self, action_name, schema, handler_func):
            self.registry[action_name] = (schema, handler_func)
    ```
*   **Extensibility**: This decouples the parser from tool execution, permitting easy additions of future tools (like `execute_python`, `git_diff`, `ripgrep`, or `web_search`) by simply decorating handler functions.

---

## 📈 4. Training Pipeline Enhancements

To support long-running, stable convergence on the RTX 4050, we will integrate these features in `trainer.py`:
1.  **Exponential Moving Average (EMA)**: Maintain a shadow copy of model weights using EMA decay ($\approx 0.999$) to stabilize validation outputs and prevent overfitting.
2.  **Early Stopping**: Automatically halt training if validation perplexity fails to improve for a consecutive window of epochs (e.g., patience of 3).
3.  **WandB/TensorBoard Integrations**: Write metrics (train loss, validation loss, tokens/sec, VRAM utilization, gradient norms) to visual dashboards.

---

## ⚡ 5. Hardware Acceleration: FlashAttention-2 Validation

*   **Method**: Our `OptimizedCausalAttention` block utilizes PyTorch's native `F.scaled_dot_product_attention` (SDPA). 
*   **Validation Test**: We will add assertions in `test_attention_optimized.py` to check that on CUDA-compatible GPUs, the dispatcher chooses the fused FlashAttention kernel backend, verifying VRAM savings.

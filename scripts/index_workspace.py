import os
import argparse
from pathlib import Path
from pocketmind.config import PocketMindConfig
from pocketmind.tokenizer.serialization import load_tokenizer
from pocketmind.retrieval.store import DocMemoryStore

def main():
    parser = argparse.ArgumentParser(description="PocketMind Workspace Indexer")
    parser.add_argument("--config", type=str, default="configs/small.yaml", help="Path to config YAML")
    parser.add_argument("--workspace", type=str, default=".", help="Root workspace to index")
    parser.add_argument("--db-path", type=str, default="data/processed/documents.sqlite3", help="Target SQLite DB path")
    args = parser.parse_args()
    
    # 1) Load config and tokenizer
    if os.path.exists(args.config):
        config = PocketMindConfig.from_yaml(args.config)
        print(f"Loaded config: {args.config}")
    else:
        config = PocketMindConfig()
        print("Using default configuration values")
        
    tokenizer_path = "data/processed/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "data/tokenizer/tokenizer.json"
        
    if not os.path.exists(tokenizer_path):
        print("Error: Tokenizer file not found. Please run train_tokenizer.py first.")
        return
        
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Loaded tokenizer with vocab size {len(tokenizer.vocab) + len(tokenizer.special_vocab)}")
    
    # 2) Initialize DocMemoryStore
    store = DocMemoryStore(args.db_path)
    print(f"Initialized database store at: {args.db_path}")
    
    # 3) Define directory exclusions
    exclude_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".gemini", "checkpoints", "runs", "data", "build", "dist"
    }
    exclude_files = {
        "uv.lock", "pyproject.toml"
    }
    allowed_extensions = {
        ".py", ".ts", ".tsx", ".md", ".json", ".yaml", ".c", ".cpp", ".h"
    }
    
    # 4) Scan and Index
    workspace_root = os.path.realpath(args.workspace)
    print(f"Scanning workspace root: {workspace_root}...")
    
    total_scanned = 0
    total_indexed = 0
    total_chunks = 0
    
    for root, dirs, files in os.walk(workspace_root):
        # Exclude directories in-place to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            suffix = Path(file).suffix
            if suffix not in allowed_extensions or file in exclude_files:
                continue
                
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, workspace_root)
            total_scanned += 1
            
            try:
                # Check binary file
                with open(full_path, "rb") as f:
                    if b"\x00" in f.read(1024):
                        continue
                        
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                chunks_inserted = store.chunk_and_index_file(
                    relative_path=rel_path,
                    content=content,
                    tokenizer=tokenizer,
                    language=suffix.lstrip(".")
                )
                
                if chunks_inserted > 0:
                    total_indexed += 1
                    total_chunks += chunks_inserted
                    print(f"Indexed: {rel_path} ({chunks_inserted} chunks)")
                    
            except Exception as e:
                print(f"Failed to index {rel_path}: {str(e)}")
                
    print("\n=== Workspace Indexing Completed ===")
    print(f"Total Files Scanned: {total_scanned}")
    print(f"Total Files Newly Indexed/Updated: {total_indexed}")
    print(f"Total Chunks Created: {total_chunks}")

if __name__ == "__main__":
    main()

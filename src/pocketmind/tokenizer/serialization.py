import json
from pathlib import Path
from pocketmind.tokenizer.bpe import BPETokenizer

def save_tokenizer(tokenizer: BPETokenizer, filepath: str | Path) -> None:
    # Convert merges key tuples to lists for JSON serialization
    serialized_merges = [
        {"p1": p[0], "p2": p[1], "id": idx}
        for p, idx in tokenizer.merges.items()
    ]
    
    data = {
        "special_tokens": tokenizer.special_tokens,
        "merges": serialized_merges
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_tokenizer(filepath: str | Path) -> BPETokenizer:
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    special_tokens = data.get("special_tokens", [])
    tokenizer = BPETokenizer(special_tokens=special_tokens)
    
    # Reconstruct merges dictionary
    merges_list = data.get("merges", [])
    
    # Sort merges by output ID to ensure dependencies are resolved sequentially
    merges_list.sort(key=lambda x: x["id"])
    
    for item in merges_list:
        p1 = item["p1"]
        p2 = item["p2"]
        merged_id = item["id"]
        
        pair = (p1, p2)
        tokenizer.merges[pair] = merged_id
        
        # Resolve bytes representations
        b1 = tokenizer.vocab[p1]
        b2 = tokenizer.vocab[p2]
        
        tokenizer.vocab[merged_id] = b1 + b2
        tokenizer.inverse_vocab[b1 + b2] = merged_id
        
    return tokenizer

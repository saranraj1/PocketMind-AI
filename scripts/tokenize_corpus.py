import argparse
import numpy as np
from pathlib import Path
from pocketmind.tokenizer.serialization import load_tokenizer

def tokenize_and_save(text_path: Path, bin_path: Path, tokenizer) -> None:
    print(f"Reading {text_path.name}...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Tokenizing {text_path.name}...")
    token_ids = tokenizer.encode(text)
    print(f"Result: {len(token_ids)} tokens.")
    
    # Verify vocab bounds (ensure no token exceeds uint16 max)
    arr = np.array(token_ids, dtype=np.uint16)
    
    print(f"Saving binary tokens to {bin_path}...")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Tokenize training corpus splits using trained BPE tokenizer.")
    parser.add_argument("--tokenizer_json", type=str, default="data/processed/tokenizer.json", help="Path to serialized tokenizer JSON.")
    parser.add_argument("--input_dir", type=str, default="data/processed", help="Directory containing train.txt and val.txt.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save train.bin and val.bin.")
    args = parser.parse_args()
    
    tokenizer_path = Path(args.tokenizer_json)
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Process train.txt -> train.bin
    train_txt = input_path / "train.txt"
    train_bin = output_path / "train.bin"
    if train_txt.exists():
        tokenize_and_save(train_txt, train_bin, tokenizer)
        
    # Process val.txt -> val.bin
    val_txt = input_path / "val.txt"
    val_bin = output_path / "val.bin"
    if val_txt.exists():
        tokenize_and_save(val_txt, val_bin, tokenizer)

if __name__ == "__main__":
    main()

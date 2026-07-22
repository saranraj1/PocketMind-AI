import os
import argparse
import urllib.request
from pathlib import Path
from pocketmind.data.cleaning import clean_text, contains_secrets
from pocketmind.data.deduplication import ExactDeduplicator, MinHashDeduplicator

WIKITEXT_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
WIKITEXT_VAL_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"

def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

def process_and_clean_file(path: Path, exact_dedup: ExactDeduplicator, minhash_dedup: MinHashDeduplicator) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {path.name}: Failed to read ({e})")
        return None
        
    # 1) Clean formatting
    cleaned = clean_text(content)
    if not cleaned.strip():
        return None
        
    # 2) Secret scanning
    if contains_secrets(cleaned):
        print(f"Skipping {path.name}: Flagged by secret scanner")
        return None
        
    # 3) Exact deduplication
    if exact_dedup.is_duplicate(cleaned):
        print(f"Skipping {path.name}: Flagged as exact duplicate")
        return None
        
    # 4) Near-deduplication
    if minhash_dedup.is_near_duplicate(cleaned, doc_id=str(path)):
        print(f"Skipping {path.name}: Flagged as near-duplicate (>85% Jaccard)")
        return None
        
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and prepare raw text corpus.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing raw source files.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed data.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exact_dedup = ExactDeduplicator()
    minhash_dedup = MinHashDeduplicator(num_perm=64, threshold=0.85)
    
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    
    if args.input_dir and os.path.exists(args.input_dir):
        # Scan and process files from custom input directory
        input_path = Path(args.input_dir)
        print(f"Scanning raw files in {input_path}...")
        
        valid_extensions = {".txt", ".py", ".md", ".json", ".yaml", ".js", ".ts", ".c", ".cpp", ".h"}
        all_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix in valid_extensions]
        
        print(f"Found {len(all_files)} files. Cleaning and deduplicating...")
        
        cleaned_docs = []
        for file_p in all_files:
            cleaned = process_and_clean_file(file_p, exact_dedup, minhash_dedup)
            if cleaned:
                cleaned_docs.append(cleaned)
                
        print(f"Processing complete. {len(cleaned_docs)} / {len(all_files)} documents preserved.")
        
        # Split documents into train (90%) and val (10%)
        split_idx = int(len(cleaned_docs) * 0.9)
        train_docs = cleaned_docs[:split_idx]
        val_docs = cleaned_docs[split_idx:]
        
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(train_docs))
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(val_docs))
            
        print(f"Saved pretraining splits to {train_path} and {val_path}")
    else:
        # Fallback: Download Wikitext-2 pretraining dataset
        print("No raw input_dir specified. Fetching Wikitext-2 pretraining dataset as fallback...")
        raw_dir = Path("data/raw")
        raw_train_file = raw_dir / "wikitext_train.txt"
        raw_val_file = raw_dir / "wikitext_val.txt"
        
        if not raw_train_file.exists():
            download_file(WIKITEXT_TRAIN_URL, raw_train_file)
        if not raw_val_file.exists():
            download_file(WIKITEXT_VAL_URL, raw_val_file)
            
        # Clean and write Wikitext files directly
        print("Cleaning fallback dataset...")
        with open(raw_train_file, "r", encoding="utf-8") as f:
            train_clean = clean_text(f.read())
        with open(raw_val_file, "r", encoding="utf-8") as f:
            val_clean = clean_text(f.read())
            
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_clean)
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_clean)
            
        print(f"Fallback splits prepared successfully at {train_path} and {val_path}")

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from pocketmind.config import PocketMindConfig
from pocketmind.tokenizer.bpe import BPETokenizer
from pocketmind.tokenizer.serialization import save_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer on prepared corpus.")
    parser.add_argument("--config", type=str, required=True, help="Path to config profile yaml.")
    parser.add_argument("--input_txt", type=str, default="data/processed/train.txt", help="Path to pretraining corpus text.")
    parser.add_argument("--output_json", type=str, default="data/processed/tokenizer.json", help="Path to output serialized tokenizer JSON.")
    args = parser.parse_args()
    
    config = PocketMindConfig.from_yaml(args.config)
    input_path = Path(args.input_txt)
    output_path = Path(args.output_json)
    
    print(f"Reading training corpus from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Initializing BPETokenizer. Target vocabulary size: {config.vocab_size}...")
    tokenizer = BPETokenizer()
    
    print("Training BPE tokenizer (this might take a few minutes)...")
    tokenizer.train(text, vocab_size=config.vocab_size, verbose=True)
    
    print(f"Saving serialized tokenizer to {output_path}...")
    save_tokenizer(tokenizer, output_path)
    print("Tokenizer training complete!")

if __name__ == "__main__":
    main()

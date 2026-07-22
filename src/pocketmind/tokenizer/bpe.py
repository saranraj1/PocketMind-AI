import re
from collections import Counter

class BPETokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        if special_tokens is None:
            special_tokens = [
                "[PAD]", "[UNK]", "[BOS]", "[EOS]",
                "[SYSTEM]", "[USER]", "[ASSISTANT]", "[ACTION]", "[OBSERVATION]"
            ]
        self.special_tokens = special_tokens
        
        # Base 256 byte mappings
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inverse_vocab = {bytes([i]): i for i in range(256)}
        
        # Special tokens mappings (offset from 256)
        self.special_vocab = {}
        self.special_inverse_vocab = {}
        for idx, token in enumerate(self.special_tokens):
            token_id = 256 + idx
            self.special_vocab[token_id] = token
            self.special_inverse_vocab[token] = token_id
            
        # Merges mapping: (id1, id2) -> merged_id
        self.merges = {}
        
        # Cache to speed up token encoding
        self.cache = {}

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        # Determine number of merge operations
        num_merges = vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return
            
        # GPT-2 style regex word splitter
        split_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+")
        words = split_pattern.findall(text)
        
        # Count raw word occurrences to optimize statistics aggregation
        word_counts = Counter(words)
        
        # Represent words as lists of base token IDs
        word_tokens = {word: list(word.encode("utf-8")) for word in word_counts}
        
        current_vocab_size = 256 + len(self.special_tokens)
        
        for merge_idx in range(num_merges):
            # Compute adjacent pair frequencies
            pair_counts = Counter()
            for word, tokens in word_tokens.items():
                freq = word_counts[word]
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i+1])] += freq
                    
            if not pair_counts:
                break
                
            # Find the most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            
            # Allocate new token ID and register merge
            new_id = current_vocab_size
            self.merges[best_pair] = new_id
            
            # Compute bytes string for the new merged token ID
            b1 = self.vocab[best_pair[0]]
            b2 = self.vocab[best_pair[1]]
            self.vocab[new_id] = b1 + b2
            self.inverse_vocab[b1 + b2] = new_id
            
            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"BPE Merge {merge_idx + 1}/{num_merges}: {best_pair} -> {new_id} (count: {best_count})")
                
            # Perform merge across all words
            new_word_tokens = {}
            for word, tokens in word_tokens.items():
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_word_tokens[word] = new_tokens
            word_tokens = new_word_tokens
            
            current_vocab_size += 1
            
        # Reset cache after training
        self.cache = {}

    def _encode_word(self, word: str) -> list[int]:
        if word in self.cache:
            return self.cache[word]
            
        tokens = list(word.encode("utf-8"))
        while len(tokens) >= 2:
            # Find all adjacent pairs
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            
            # Select the pair with the lowest merge index priority
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
                
            # Merge best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(self.merges[best_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        self.cache[word] = tokens
        return tokens

    def _encode_text_only(self, text: str) -> list[int]:
        split_pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+")
        words = split_pattern.findall(text)
        ids = []
        for word in words:
            ids.extend(self._encode_word(word))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_inverse_vocab:
            return self._encode_text_only(text)
            
        # Split text by special tokens pattern
        escaped_specials = [re.escape(t) for t in self.special_inverse_vocab]
        special_pattern = re.compile(f"({'|'.join(escaped_specials)})")
        
        parts = special_pattern.split(text)
        ids = []
        for part in parts:
            if part in self.special_inverse_vocab:
                ids.append(self.special_inverse_vocab[part])
            else:
                ids.extend(self._encode_text_only(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            elif token_id in self.special_vocab:
                byte_parts.append(self.special_vocab[token_id].encode("utf-8"))
        return b"".join(byte_parts).decode("utf-8", errors="replace")

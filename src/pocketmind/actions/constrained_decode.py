import torch
from pocketmind.tokenizer.bpe import BPETokenizer

class JSONGrammarParser:
    def __init__(self):
        self.stack = []
        self.in_string = False
        self.escaped = False
        self.last_char = ""

    def copy(self) -> "JSONGrammarParser":
        p = JSONGrammarParser()
        p.stack = list(self.stack)
        p.in_string = self.in_string
        p.escaped = self.escaped
        p.last_char = self.last_char
        return p

    def consume(self, char: str) -> bool:
        if self.in_string:
            if self.escaped:
                self.escaped = False
                return True
            if char == "\\":
                self.escaped = True
                return True
            if char == '"':
                self.in_string = False
                return True
            # Control characters (ASCII 0-31) must be escaped in JSON
            if ord(char) < 32:
                return False
            return True

        # Outside of string
        if char.isspace():
            return True

        if char == '"':
            self.in_string = True
            return True

        if char == "{":
            self.stack.append("}")
            return True

        if char == "[":
            self.stack.append("]")
            return True

        if char in ("}", "]"):
            if not self.stack or self.stack[-1] != char:
                return False
            self.stack.pop()
            return True

        if char in (":", ","):
            return True

        # Valid characters for numbers, booleans, or null
        if char in "0123456789.-tfn":
            return True

        return False

    def consume_string(self, s: str) -> bool:
        for char in s:
            if not self.consume(char):
                return False
        return True

def get_allowed_tokens_mask(
    prefix: str,
    tokenizer: BPETokenizer,
    parser_state: JSONGrammarParser
) -> torch.Tensor:
    # Get total vocabulary size (base + special)
    vocab_size = 256 + len(tokenizer.special_tokens) + len(tokenizer.merges)
    mask = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
    
    # Check each token ID
    for token_id in range(vocab_size):
        # Determine string representation of token ID
        if token_id in tokenizer.vocab:
            val = tokenizer.vocab[token_id]
            try:
                # Decode byte sequence
                val_str = val.decode("utf-8")
            except UnicodeDecodeError:
                # Reject invalid UTF-8 token segments during JSON string transitions
                continue
        elif token_id in tokenizer.special_vocab:
            # Special tokens are handled separately or allowed inside JSON strings
            val_str = tokenizer.special_vocab[token_id]
        else:
            continue
            
        # Test character transition sequence on a cloned parser state
        clone = parser_state.copy()
        if clone.consume_string(val_str):
            mask[token_id] = 0.0  # Set logit mask offset to 0 (allowed)
            
    return mask

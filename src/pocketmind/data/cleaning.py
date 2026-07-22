import re
import math
from collections import Counter

# Common secret regex patterns (GitHub tokens, OpenAI keys, AWS keys, etc.)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{24,}", re.IGNORECASE),                         # OpenAI style
    re.compile(r"ghp_[a-zA-Z0-9]{36,}", re.IGNORECASE),                         # GitHub Personal Access Token
    re.compile(r"amzn\.mws\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE), # Amazon MWS
    re.compile(r"AKIA[0-9A-Z]{16}", re.IGNORECASE),                             # AWS Access Key ID
    re.compile(r"private_key|client_secret|api_key|password|db_password", re.IGNORECASE) # General metadata keys
]

def calculate_entropy(s: str) -> float:
    # Compute Shannon entropy of string
    if not s:
        return 0.0
    entropy = 0.0
    length = len(s)
    counts = Counter(s)
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def contains_secrets(text: str, entropy_threshold: float = 4.5, min_len: int = 16) -> bool:
    # 1) Check regex patterns
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
            
    # 2) Scan words/tokens for high Shannon entropy (often indicates keys/hashes)
    words = re.split(r"\s+", text)
    for word in words:
        if len(word) >= min_len:
            entropy = calculate_entropy(word)
            # If word is a single continuous high-entropy token (no separators), flag it
            if entropy >= entropy_threshold and re.match(r"^[a-zA-Z0-9/+=_-]+$", word):
                return True
                
    return False

def clean_text(text: str) -> str:
    # 1) Standardize line endings to \n and remove \r
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 2) Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    
    # 3) Collapse excessive consecutive empty lines (limit to max 2 consecutive blank lines)
    collapsed_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                collapsed_lines.append(line)
        else:
            blank_count = 0
            collapsed_lines.append(line)
            
    return "\n".join(collapsed_lines)

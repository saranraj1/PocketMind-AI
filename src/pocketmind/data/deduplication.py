import binascii
import hashlib
import random
from typing import Set, List

class ExactDeduplicator:
    def __init__(self):
        self.hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        # Calculate SHA-256 hash of the cleaned text representation
        hasher = hashlib.sha256()
        hasher.update(text.encode("utf-8"))
        h = hasher.hexdigest()
        
        if h in self.hashes:
            return True
            
        self.hashes.add(h)
        return False

class MinHashDeduplicator:
    def __init__(self, num_perm: int = 64, threshold: float = 0.85, n_gram: int = 5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_gram = n_gram
        self.prime = 2147483647  # Mersenne Prime 2^31 - 1
        
        # Pre-generate coefficients using a fixed random seed
        rng = random.Random(1337)
        self.a = [rng.randint(1, self.prime - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.prime - 1) for _ in range(num_perm)]
        
        # Saved signatures list: list of tuples (doc_id, signature_list)
        self.signatures: List[tuple[str, List[int]]] = []

    def _get_shingles(self, text: str) -> Set[str]:
        # Extract character n-grams as shingles
        shingles = set()
        if len(text) < self.n_gram:
            shingles.add(text)
            return shingles
            
        for i in range(len(text) - self.n_gram + 1):
            shingles.add(text[i : i + self.n_gram])
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm
            
        # Initialize signature values to infinity
        sig = [self.prime] * self.num_perm
        
        for shingle in shingles:
            # Generate a stable 32-bit integer hash from the shingle bytes
            x = binascii.crc32(shingle.encode("utf-8")) & 0xffffffff
            for i in range(self.num_perm):
                # Hash function formula: h_i(x) = (a_i * x + b_i) % prime
                h_val = (self.a[i] * x + self.b[i]) % self.prime
                if h_val < sig[i]:
                    sig[i] = h_val
                    
        return sig

    def is_near_duplicate(self, text: str, doc_id: str = "") -> bool:
        new_sig = self.compute_signature(text)
        
        for saved_id, saved_sig in self.signatures:
            # Compute estimated Jaccard similarity: matches / total
            matches = sum(1 for i in range(self.num_perm) if new_sig[i] == saved_sig[i])
            similarity = matches / self.num_perm
            if similarity >= self.threshold:
                return True
                
        # Register new signature if not matching any duplicate
        self.signatures.append((doc_id, new_sig))
        return False

from pocketmind.data.cleaning import clean_text, contains_secrets
from pocketmind.data.deduplication import ExactDeduplicator, MinHashDeduplicator

def test_text_cleaning_formatting():
    dirty = "line1 \r\nline2   \n\n\n\nline3"
    cleaned = clean_text(dirty)
    # Trailing whitespaces stripped, \r removed, blanks collapsed to max 2
    assert cleaned == "line1\nline2\n\n\nline3"

def test_secret_scanner():
    clean_code = "def add(a, b):\n    return a + b"
    secret_code = "sk-proj-1234567890abcdef1234567890abcdef12345678"
    high_entropy_token = "abcXYZ1234567890/+_=-defXYZ" # random high entropy token
    
    assert not contains_secrets(clean_code)
    assert contains_secrets(secret_code)
    assert contains_secrets(high_entropy_token, entropy_threshold=4.0, min_len=12)

def test_exact_dedup():
    dedup = ExactDeduplicator()
    text = "unique pretraining doc"
    assert not dedup.is_duplicate(text)
    assert dedup.is_duplicate(text)

def test_minhash_near_dedup():
    dedup = MinHashDeduplicator(num_perm=64, threshold=0.8)
    
    text1 = "This is a clean and unique sample paragraph for MinHash testing."
    text2 = "This is a clean and unique sample paragraph for MinHash testing. (modified)" # highly similar
    text3 = "Completely different text content with zero overlap in n-grams." # distinct
    
    assert not dedup.is_near_duplicate(text1, "doc1")
    assert dedup.is_near_duplicate(text2, "doc2") # should be flagged as near-duplicate
    assert not dedup.is_near_duplicate(text3, "doc3") # should be accepted

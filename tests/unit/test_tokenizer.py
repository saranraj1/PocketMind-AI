import tempfile
import os
from pocketmind.tokenizer.bpe import BPETokenizer
from pocketmind.tokenizer.serialization import save_tokenizer, load_tokenizer

def test_bpe_training_lossless():
    text = "hello world! hello world! hello there. welcome to pocketmind."
    tokenizer = BPETokenizer()
    # Train BPE to a small vocab size (e.g. 270)
    tokenizer.train(text, vocab_size=270, verbose=False)
    
    # 1) Verify all characters are recoverable (lossless loop)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
    
    # 2) Verify special tokens are correctly encoded
    special_text = "[SYSTEM] Hello [USER] how are you? [ASSISTANT]"
    encoded_special = tokenizer.encode(special_text)
    decoded_special = tokenizer.decode(encoded_special)
    assert decoded_special == special_text
    
    # Check that special token IDs are actually present
    system_id = tokenizer.special_inverse_vocab["[SYSTEM]"]
    assert system_id in encoded_special

def test_tokenizer_serialization():
    text = "first principles LLM tokenizer training verify rules."
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=275, verbose=False)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_name = tmp.name
        
    try:
        # Save tokenizer
        save_tokenizer(tokenizer, tmp_name)
        
        # Load tokenizer
        reloaded = load_tokenizer(tmp_name)
        
        # Verify merges and vocab match
        assert len(tokenizer.merges) == len(reloaded.merges)
        assert len(tokenizer.vocab) == len(reloaded.vocab)
        
        # Verify encoding output is identical
        assert tokenizer.encode(text) == reloaded.encode(text)
        
        # Verify special tokens are preserved
        assert tokenizer.special_inverse_vocab == reloaded.special_inverse_vocab
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

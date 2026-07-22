import os
import tempfile
import pytest
from pocketmind.tokenizer.bpe import BPETokenizer
from pocketmind.retrieval.store import DocMemoryStore

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_docs.sqlite3")
        yield db_path

@pytest.fixture
def mock_tokenizer():
    # Simple byte tokenizer mockup to count characters/tokens for test chunking
    tokenizer = BPETokenizer()
    tokenizer.vocab = {i: bytes([i]) for i in range(256)}
    tokenizer.special_vocab = {256: "<PAD>"}
    tokenizer.merges = {}
    return tokenizer

def test_store_initialization(temp_db):
    store = DocMemoryStore(temp_db)
    assert os.path.exists(temp_db)
    
    # Check tables can be queried
    with store._get_conn() as conn:
        res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = {row["name"] for row in res}
        assert "documents" in table_names
        assert "chunks" in table_names
        assert "chunks_fts" in table_names
        assert "corrections" in table_names

def test_store_chunking_overlap(temp_db, mock_tokenizer):
    store = DocMemoryStore(temp_db)
    
    # Create text with 20 distinct lines
    lines = [f"This is distinct line number {i} with some extra padding text." for i in range(20)]
    text = "\n".join(lines)
    
    # Chunk generation with small targets to force splitting
    chunks = store._generate_chunks(text, mock_tokenizer, target_tokens=100, overlap_tokens=20)
    
    assert len(chunks) > 1
    # Check start/end line bounds are ascending and correct
    for idx, (chunk_text, start, end, tokens) in enumerate(chunks):
        assert start >= 1
        assert end >= start
        assert tokens > 0
        assert chunk_text.startswith(lines[start - 1])
        assert chunk_text.endswith(lines[end - 1])

def test_store_bm25_retrieval(temp_db, mock_tokenizer):
    store = DocMemoryStore(temp_db)
    
    doc1_content = "This document discusses antigravity physics and deep space propulsion concepts."
    doc2_content = "Here is a delicious recipe for baking banana bread with walnuts and cinnamon."
    doc3_content = "Quantum superposition is a fundamental principle of quantum mechanics."
    
    store.chunk_and_index_file("physics/space.txt", doc1_content, mock_tokenizer)
    store.chunk_and_index_file("recipes/banana.txt", doc2_content, mock_tokenizer)
    store.chunk_and_index_file("quantum/mechanics.txt", doc3_content, mock_tokenizer)
    
    # 1) Search for "banana"
    results_banana = store.search("banana bread")
    assert len(results_banana) >= 1
    assert results_banana[0]["source_path"] == "recipes/banana.txt"
    assert "banana bread" in results_banana[0]["content"]
    assert results_banana[0]["bm25_score"] is not None
    
    # 2) Search for "superposition"
    results_quantum = store.search("quantum superposition")
    assert len(results_quantum) >= 1
    assert results_quantum[0]["source_path"] == "quantum/mechanics.txt"
    assert "superposition" in results_quantum[0]["content"]
    
    # 3) Search for non-existent keyword
    results_none = store.search("unobtainium")
    assert len(results_none) == 0

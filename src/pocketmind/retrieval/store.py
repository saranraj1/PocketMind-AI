import os
import re
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from pocketmind.tokenizer.bpe import BPETokenizer

class DocMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            # 1) documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    language TEXT,
                    indexed_at TEXT NOT NULL,
                    UNIQUE(source_path)
                );
            """)
            
            # 2) chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE
                );
            """)
            
            # 3) FTS5 virtual table
            # Check if FTS5 is available by running a test creation
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content,
                        content_rowid='id'
                    );
                """)
                
                # Triggers to keep chunks_fts in sync with chunks table
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
                    END;
                """)
            except sqlite3.OperationalError:
                # Fallback to standard virtual table or mock if FTS5 not compiled in SQLite (extremely rare)
                raise RuntimeError("FTS5 extension is required in SQLite for PocketMind search capabilities.")

            # 4) corrections table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request TEXT NOT NULL,
                    context TEXT,
                    incorrect_output TEXT,
                    corrected_output TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    training_status TEXT NOT NULL DEFAULT 'pending'
                );
            """)
            conn.commit()

    def chunk_and_index_file(
        self,
        relative_path: str,
        content: str,
        tokenizer: BPETokenizer,
        language: str | None = None
    ) -> int:
        # Calculate overall file hash
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        
        # Check if already indexed and unchanged
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, source_hash FROM documents WHERE source_path = ?",
                (relative_path,)
            )
            row = cur.fetchone()
            if row:
                if row["source_hash"] == file_hash:
                    # File exists and is unmodified: skip re-indexing
                    return 0
                else:
                    # File modified: remove old version (triggers ON DELETE CASCADE)
                    cur.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            
            # Register document
            cur.execute(
                """
                INSERT INTO documents (source_path, source_hash, language, indexed_at)
                VALUES (?, ?, ?, ?)
                """,
                (relative_path, file_hash, language, datetime.now(timezone.utc).isoformat())
            )
            doc_id = cur.lastrowid
            
            # Slice content into chunks of ~200-400 BPE tokens with 30-60 overlap tokens
            chunks = self._generate_chunks(content, tokenizer)
            
            # Insert chunks
            inserted_count = 0
            for idx, (chunk_text, start_l, end_l, tokens_num) in enumerate(chunks):
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                try:
                    cur.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, start_line, end_line, content, token_count, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (doc_id, idx, start_l, end_l, chunk_text, tokens_num, chunk_hash)
                    )
                    inserted_count += 1
                except sqlite3.IntegrityError:
                    # Duplicate chunk text content: skip chunk insertion
                    continue
            conn.commit()
            
        return inserted_count

    def _generate_chunks(
        self,
        content: str,
        tokenizer: BPETokenizer,
        target_tokens: int = 300,
        overlap_tokens: int = 40
    ) -> list[tuple[str, int, int, int]]:
        lines = content.splitlines()
        chunks = []
        
        # Cache line tokens count
        line_tokens = []
        for line in lines:
            line_tokens.append(len(tokenizer.encode(line + "\n")))
            
        start_idx = 0
        total_lines = len(lines)
        
        while start_idx < total_lines:
            curr_tokens = 0
            end_idx = start_idx
            
            # Consume lines up to target_tokens
            while end_idx < total_lines:
                tokens_in_line = line_tokens[end_idx]
                if curr_tokens + tokens_in_line > target_tokens + 50 and curr_tokens > 0:
                    break
                curr_tokens += tokens_in_line
                end_idx += 1
                
            # Grab chunk text
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines)
            
            # Guard against empty chunk
            if chunk_text.strip():
                chunks.append((chunk_text, start_idx + 1, end_idx, curr_tokens))
                
            # If we reached the end, terminate
            if end_idx >= total_lines:
                break
                
            # Backtrack start_idx for overlap
            overlap_tokens_accum = 0
            backtrack_idx = end_idx - 1
            while backtrack_idx > start_idx and overlap_tokens_accum + line_tokens[backtrack_idx] < overlap_tokens:
                overlap_tokens_accum += line_tokens[backtrack_idx]
                backtrack_idx -= 1
                
            # Ensure progress
            start_idx = max(end_idx, backtrack_idx)
            
        return chunks

    def search(self, query_text: str, limit: int = 5) -> list[dict]:
        results = []
        # SQLite FTS5 MATCH queries don't like empty or only special characters
        clean_query = re.sub(r'[^\w\s]', ' ', query_text).strip()
        if not clean_query:
            return []
            
        # Join words with OR for BM25 match flexibility
        match_expr = " OR ".join(clean_query.split())
        
        with self._get_conn() as conn:
            cur = conn.cursor()
            try:
                # Execute FTS5 ranking query
                cur.execute(
                    """
                    SELECT c.id, c.document_id, c.chunk_index, c.start_line, c.end_line, c.content, c.token_count, d.source_path, fts.rank
                    FROM chunks_fts fts
                    JOIN chunks c ON fts.rowid = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY fts.rank ASC
                    LIMIT ?
                    """,
                    (match_expr, limit)
                )
                for row in cur.fetchall():
                    results.append({
                        "id": row["id"],
                        "source_path": row["source_path"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "content": row["content"],
                        "token_count": row["token_count"],
                        "bm25_score": -row["rank"]  # invert rank since lower is better in FTS5 BM25
                    })
            except sqlite3.OperationalError:
                # Fallback to LIKE if MATCH fails syntax check
                like_pattern = f"%{clean_query}%"
                cur.execute(
                    """
                    SELECT c.id, c.document_id, c.chunk_index, c.start_line, c.end_line, c.content, c.token_count, d.source_path
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.content LIKE ?
                    LIMIT ?
                    """,
                    (like_pattern, limit)
                )
                for row in cur.fetchall():
                    results.append({
                        "id": row["id"],
                        "source_path": row["source_path"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "content": row["content"],
                        "token_count": row["token_count"],
                        "bm25_score": 1.0
                    })
        return results

    def add_correction(
        self,
        request: str,
        context: str | None,
        incorrect_output: str | None,
        corrected_output: str
    ) -> int:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO corrections (request, context, incorrect_output, corrected_output, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request, context, incorrect_output, corrected_output, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
            return cur.lastrowid
            
    def get_pending_corrections(self) -> list[dict]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM corrections WHERE training_status = 'pending'")
            return [dict(row) for row in cur.fetchall()]

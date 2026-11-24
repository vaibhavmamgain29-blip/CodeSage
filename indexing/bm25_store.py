import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Plus, BM25Okapi
from codesage.ingestion.chunker import CodeChunk


def code_tokenize(text: str) -> List[str]:
    """
    Code-aware tokenizer:
    1. Splits camelCase (e.g. 'getUserProfile' -> 'get', 'User', 'Profile')
    2. Splits snake_case and special characters
    3. Keeps original identifiers (for exact matching) as well as subwords
    """
    if not text:
        return []

    # Preserve original tokens before splitting
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)
    tokens: List[str] = []

    for t in raw_tokens:
        lower_t = t.lower()
        tokens.append(lower_t)

        # Split camelCase
        camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", t).split()
        if len(camel_split) > 1:
            for sub in camel_split:
                tokens.append(sub.lower())

        # Split snake_case
        if "_" in t:
            for sub in t.split("_"):
                if sub:
                    tokens.append(sub.lower())

    return tokens


class BM25Index:
    """Keyword search index over code chunks using BM25."""

    def __init__(self):
        self.chunks: List[CodeChunk] = []
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    def build(self, chunks: List[CodeChunk]) -> None:
        """Build BM25 index from code chunks."""
        self.chunks = chunks
        self.chunk_ids = [c.id for c in chunks]
        self.corpus_tokens = []

        for chunk in chunks:
            # Combine symbol name, docstring, imports, and content for keyword search
            text_to_index = f"{chunk.symbol_name} {chunk.symbol_name} {chunk.docstring or ''} {' '.join(chunk.imports)} {chunk.content}"
            tokens = code_tokenize(text_to_index)
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            self.bm25 = BM25Plus(self.corpus_tokens)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search BM25 index and return (chunk, score) pairs."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = code_tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    def save(self, filepath: Path) -> None:
        """Persist BM25 index to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens,
                "bm25": self.bm25
            }, f)

    @classmethod
    def load(cls, filepath: Path) -> "BM25Index":
        """Load BM25 index from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index not found at {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.chunks = data["chunks"]
        instance.chunk_ids = data["chunk_ids"]
        instance.corpus_tokens = data["corpus_tokens"]
        instance.bm25 = data["bm25"]
        return instance

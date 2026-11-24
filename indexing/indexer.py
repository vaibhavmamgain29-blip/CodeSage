import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from codesage.ingestion.cloner import Cloner, walk_code_files
from codesage.ingestion.chunker import chunk_file, CodeChunk
from codesage.indexing.bm25_store import BM25Index
from codesage.indexing.vector_store import VectorStore
from codesage.config import settings


class RepoIndexer:
    """Coordinates repository cloning, file traversal, AST chunking, vector embedding, and BM25 indexing."""

    def __init__(
        self,
        repo_url_or_path: str,
        persist_dir: Path = settings.chroma_persist_dir,
        cache_dir: Path = settings.repos_cache_dir
    ):
        self.cloner = Cloner(cache_dir=cache_dir)
        self.repo_path = self.cloner.resolve_repo_path(repo_url_or_path)
        self.repo_name = self.repo_path.name
        self.persist_dir = Path(persist_dir)

        # Namespaced storage per repository
        self.vector_store = VectorStore(
            persist_dir=self.persist_dir,
            collection_name=f"repo_{self.repo_name.replace('-', '_').replace('.', '_')}"
        )
        self.bm25_index = BM25Index()
        self.bm25_path = self.persist_dir / f"bm25_{self.repo_name}.pkl"
        self.chunks: List[CodeChunk] = []

    def index(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Index the repository end-to-end:
        1. Discover code files
        2. Chunk with AST (or fallback)
        3. Index in Chroma vector store
        4. Index in BM25 store and persist
        """
        start_time = time.time()

        files_to_index = list(walk_code_files(self.repo_path))
        total_files = len(files_to_index)

        all_chunks: List[CodeChunk] = []
        for idx, (abs_path, rel_path) in enumerate(files_to_index, start=1):
            if progress_callback:
                progress_callback(idx / max(total_files, 1), f"Parsing ({idx}/{total_files}): {rel_path}")

            file_chunks = chunk_file(abs_path, rel_path)
            all_chunks.extend(file_chunks)

        self.chunks = all_chunks

        # Index Vector store
        if progress_callback:
            progress_callback(0.8, f"Generating embeddings for {len(all_chunks)} chunks...")
        self.vector_store.clear()
        self.vector_store.add_chunks(all_chunks)

        # Index BM25 store
        if progress_callback:
            progress_callback(0.95, "Building BM25 keyword index...")
        self.bm25_index.build(all_chunks)
        self.bm25_index.save(self.bm25_path)

        if progress_callback:
            progress_callback(1.0, "Indexing complete!")

        elapsed = time.time() - start_time
        return {
            "repo_name": self.repo_name,
            "repo_path": str(self.repo_path),
            "files_indexed": total_files,
            "chunks_created": len(all_chunks),
            "elapsed_sec": round(elapsed, 2)
        }

    def load_existing_index(self) -> bool:
        """Attempt to load persisted BM25 and vector store for this repo without re-indexing."""
        if self.bm25_path.exists():
            try:
                self.bm25_index = BM25Index.load(self.bm25_path)
                self.chunks = self.bm25_index.chunks
                return True
            except Exception:
                return False
        return False

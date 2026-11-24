from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from codesage.ingestion.chunker import CodeChunk
from codesage.indexing.embedder import Embedder
from codesage.config import settings


class VectorStore:
    """Chroma-based vector store for code chunks."""

    def __init__(self, persist_dir: Path = settings.chroma_persist_dir, collection_name: str = "codesage_chunks"):
        import chromadb
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = Embedder(settings.embedding_model)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        """Embed and store chunks in Chroma."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        texts = [f"{c.file_path} {c.symbol_name}\n{c.content}" for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "imports": ",".join(c.imports),
                "docstring": c.docstring or "",
            }
            for c in chunks
        ]

        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)

        # Upsert in Chroma in batches of 100
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=[c.content for c in chunks[i:end_idx]],
                metadatas=metadatas[i:end_idx]
            )

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Query vector store and return list of (CodeChunk, cosine_similarity_score)."""
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["metadatas", "documents", "distances"]
        )

        output: List[Tuple[CodeChunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma cosine distance = 1 - cosine_similarity
            score = max(0.0, 1.0 - dist)
            chunk = CodeChunk(
                id=chunk_id,
                file_path=meta["file_path"],
                language=meta["language"],
                symbol_name=meta["symbol_name"],
                symbol_type=meta["symbol_type"],
                start_line=int(meta["start_line"]),
                end_line=int(meta["end_line"]),
                content=doc,
                imports=meta["imports"].split(",") if meta.get("imports") else [],
                docstring=meta.get("docstring") or None
            )
            output.append((chunk, float(score)))

        return output

    def clear(self) -> None:
        """Reset the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

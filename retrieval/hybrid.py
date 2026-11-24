from typing import List, Tuple, Dict, Any, Optional
from codesage.ingestion.chunker import CodeChunk
from codesage.config import settings


def reciprocal_rank_fusion(
    vector_results: List[Tuple[CodeChunk, float]],
    bm25_results: List[Tuple[CodeChunk, float]],
    rrf_k: int = 60,
    top_k: int = 5
) -> List[Tuple[CodeChunk, float]]:
    """
    Combines vector search and BM25 search rankings using Reciprocal Rank Fusion (RRF).
    RRF score for doc d: sum(1 / (k + rank_i(d))) for each system i where d appears.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, CodeChunk] = {}

    # Process vector rankings (rank is 1-indexed)
    for rank, (chunk, _) in enumerate(vector_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_map[chunk.id] = chunk

    # Process BM25 rankings (rank is 1-indexed)
    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (rrf_k + rank))
        chunk_map[chunk.id] = chunk

    # Sort chunks by descending RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids[:top_k]]


class HybridRetriever:
    """Retriever supporting 'vector', 'bm25', and 'hybrid' modes."""

    def __init__(self, vector_store: Any, bm25_index: Any):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Tuple[CodeChunk, float]]:
        """Retrieve chunks based on selected mode."""
        mode = mode.lower()

        if mode == "vector":
            return self.vector_store.query(query, top_k=top_k)
        elif mode == "bm25":
            return self.bm25_index.query(query, top_k=top_k)
        elif mode == "hybrid":
            # For hybrid, retrieve more candidates from each system before fusion
            candidate_k = max(top_k * 3, 15)
            vec_res = self.vector_store.query(query, top_k=candidate_k)
            bm25_res = self.bm25_index.query(query, top_k=candidate_k)
            return reciprocal_rank_fusion(vec_res, bm25_res, rrf_k=rrf_k, top_k=top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Must be 'vector', 'bm25', or 'hybrid'.")

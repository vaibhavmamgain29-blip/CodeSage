import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from codesage.ingestion.chunker import CodeChunk
from codesage.retrieval.hybrid import HybridRetriever
from codesage.qa.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT
from codesage.llm.client import LLMClient
from codesage.config import settings


@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    raw_citation: str
    is_valid: bool = False
    matching_chunk: Optional[CodeChunk] = None


@dataclass
class QAResponse:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[Tuple[CodeChunk, float]] = field(default_factory=list)
    mode: str = "hybrid"
    latency_sec: float = 0.0


CITATION_PATTERN = re.compile(r"([a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+):(\d+)(?:-(\d+))?")


class QAService:
    """Answers developer questions about a codebase with verified citations."""

    def __init__(self, retriever: HybridRetriever, llm_client: Optional[LLMClient] = None):
        self.retriever = retriever
        self.llm = llm_client or LLMClient()

    def ask(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> QAResponse:
        """Answer question with citations and verification."""
        retrieved = self.retriever.retrieve(query=question, mode=mode, top_k=top_k)

        if not retrieved:
            return QAResponse(
                question=question,
                answer="Not found in the repository: no matching code chunks could be retrieved.",
                citations=[],
                retrieved_chunks=[],
                mode=mode
            )

        # Build context
        context_blocks = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            block = (
                f"--- Chunk {rank} | {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
                f"({chunk.symbol_type}: {chunk.symbol_name}) ---\n"
                f"{chunk.content}\n"
            )
            context_blocks.append(block)

        formatted_context = "\n".join(context_blocks)
        user_prompt = QA_USER_PROMPT.format(context=formatted_context, question=question)

        llm_resp = self.llm.generate(prompt=user_prompt, system_instruction=QA_SYSTEM_PROMPT)

        raw_answer = llm_resp.content.strip()

        # Extract and verify citations
        citations = self._extract_and_verify_citations(raw_answer, [c for c, _ in retrieved])

        return QAResponse(
            question=question,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved,
            mode=mode,
            latency_sec=llm_resp.latency_sec
        )

    def _extract_and_verify_citations(
        self,
        answer_text: str,
        indexed_chunks: List[CodeChunk]
    ) -> List[Citation]:
        """
        Finds citations formatted as path:start-end in the text, and verifies
        whether they correspond to actual indexed chunks.
        """
        citations: List[Citation] = []
        seen: Set[str] = set()

        matches = CITATION_PATTERN.findall(answer_text)
        for path, start_s, end_s in matches:
            start_l = int(start_s)
            end_l = int(end_s) if end_s else start_l
            raw_str = f"{path}:{start_l}-{end_l}" if end_s else f"{path}:{start_l}"

            if raw_str in seen:
                continue
            seen.add(raw_str)

            norm_path = path.replace("\\", "/").strip("./")

            # Check validity against retrieved and indexed chunks
            matching_chunk = None
            is_valid = False

            for chunk in indexed_chunks:
                chunk_norm_path = chunk.file_path.replace("\\", "/").strip("./")
                if norm_path == chunk_norm_path or norm_path.endswith(chunk_norm_path):
                    # Check line range overlap or proximity
                    if not (end_l < chunk.start_line or start_l > chunk.end_line):
                        is_valid = True
                        matching_chunk = chunk
                        break

            citations.append(Citation(
                file_path=path,
                start_line=start_l,
                end_line=end_l,
                raw_citation=raw_str,
                is_valid=is_valid,
                matching_chunk=matching_chunk
            ))

        return citations

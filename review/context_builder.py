from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from codesage.review.diff_parser import FileDiff, DiffHunk
from codesage.ingestion.chunker import CodeChunk, chunk_file


class ContextBuilder:
    """Attaches full enclosing functions or classes to diff hunks for richer context."""

    def __init__(self, repo_path: Optional[Path] = None, chunks: Optional[List[CodeChunk]] = None):
        self.repo_path = Path(repo_path) if repo_path else None
        self.chunks_by_file: Dict[str, List[CodeChunk]] = {}

        if chunks:
            for c in chunks:
                self.chunks_by_file.setdefault(c.file_path, []).append(c)

    def _get_file_chunks(self, rel_path: str) -> List[CodeChunk]:
        if rel_path in self.chunks_by_file:
            return self.chunks_by_file[rel_path]

        if self.repo_path:
            full_path = self.repo_path / rel_path
            if full_path.exists():
                file_chunks = chunk_file(full_path, rel_path)
                self.chunks_by_file[rel_path] = file_chunks
                return file_chunks

        return []

    def build_file_context(self, file_diff: FileDiff) -> Dict[str, Any]:
        """
        Build contextual representation of a file diff.
        Finds enclosing functions/classes for each changed hunk.
        """
        rel_path = file_diff.target_path
        file_chunks = self._get_file_chunks(rel_path)

        hunk_contexts = []
        for hunk in file_diff.hunks:
            enclosing_chunks = []
            changed_lines = set(hunk.added_lines)

            # Match changed lines with enclosing chunk (function / method / class)
            for chunk in file_chunks:
                # Check overlap
                if any(chunk.start_line <= line <= chunk.end_line for line in changed_lines):
                    enclosing_chunks.append({
                        "symbol_name": chunk.symbol_name,
                        "symbol_type": chunk.symbol_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "content": chunk.content
                    })

            hunk_contexts.append({
                "hunk_header": hunk.header,
                "diff_lines": "\n".join(hunk.annotated_lines or hunk.lines),
                "enclosing_symbols": enclosing_chunks
            })

        return {
            "file": rel_path,
            "is_new": file_diff.is_new,
            "is_deleted": file_diff.is_deleted,
            "hunks": hunk_contexts
        }

    def format_diff_for_prompt(self, file_diff: FileDiff) -> str:
        """Format a single file diff and its enclosing contexts for the LLM review prompt."""
        ctx = self.build_file_context(file_diff)
        output = [f"### File: {ctx['file']}"]

        if ctx["is_new"]:
            output.append("(New File)")
        if ctx["is_deleted"]:
            output.append("(Deleted File)")

        for idx, h in enumerate(ctx["hunks"], start=1):
            output.append(f"\n--- Hunk {idx}: {h['hunk_header']} ---")
            output.append("```diff")
            output.append(h["diff_lines"])
            output.append("```")

            if h["enclosing_symbols"]:
                output.append("\nEnclosing Code Context:")
                for sym in h["enclosing_symbols"]:
                    output.append(f"**{sym['symbol_type'].capitalize()}: {sym['symbol_name']} (lines {sym['start_line']}-{sym['end_line']})**")
                    output.append("```python")
                    output.append(sym["content"])
                    output.append("```")

        return "\n".join(output)

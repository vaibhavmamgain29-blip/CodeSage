import ast
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class CodeChunk:
    id: str  # e.g. "path/to/file.py::ClassName.method_name"
    file_path: str  # Relative path in repo
    language: str  # "python"
    symbol_name: str  # Name of function, method, class, or chunk
    symbol_type: str  # "function" | "method" | "class" | "module" | "window"
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    content: str  # Source code of the chunk
    imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_start_line(node: ast.AST) -> int:
    """Return start line including decorator lines if any."""
    start = getattr(node, "lineno", 1)
    if hasattr(node, "decorator_list") and node.decorator_list:
        decorator_lines = [d.lineno for d in node.decorator_list if hasattr(d, "lineno")]
        if decorator_lines:
            start = min(start, min(decorator_lines))
    return start


def _get_end_line(node: ast.AST, default: int) -> int:
    """Return end line of AST node."""
    return getattr(node, "end_lineno", default)


def _extract_imports(tree: ast.AST) -> List[str]:
    """Collect imported module and symbol names from AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)
    return sorted(list(set(imports)))


def _chunk_by_line_window(file_content: str, rel_path: str, window_size: int = 50, overlap: int = 10) -> List[CodeChunk]:
    """Fallback chunker for unparsable or non-AST files."""
    lines = file_content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return []

    chunks: List[CodeChunk] = []
    step = max(1, window_size - overlap)

    idx = 0
    for start_idx in range(0, total_lines, step):
        end_idx = min(start_idx + window_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        chunk_content = "\n".join(chunk_lines)

        start_line = start_idx + 1
        end_line = end_idx
        chunk_id = f"{rel_path}::lines_{start_line}_{end_line}"

        chunks.append(CodeChunk(
            id=chunk_id,
            file_path=rel_path,
            language="python",
            symbol_name=f"lines_{start_line}_{end_line}",
            symbol_type="window",
            start_line=start_line,
            end_line=end_line,
            content=chunk_content,
            imports=[],
            docstring=None
        ))
        idx += 1
        if end_idx >= total_lines:
            break

    return chunks


def chunk_python_code(file_content: str, rel_path: str) -> List[CodeChunk]:
    """
    Parse Python code into function, method, and class chunks with metadata using ast.
    Falls back to line-window chunking if syntax error occurs.
    """
    if not file_content.strip():
        return []

    lines = file_content.splitlines()
    total_lines = len(lines)

    try:
        tree = ast.parse(file_content, filename=rel_path)
    except Exception:
        # Syntax error or unparsable Python file -> fallback to line-window
        return _chunk_by_line_window(file_content, rel_path)

    global_imports = _extract_imports(tree)
    chunks: List[CodeChunk] = []

    # Check for top-level module docstring
    module_docstring = ast.get_docstring(tree)

    class CodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class: Optional[str] = None

        def visit_ClassDef(self, node: ast.ClassDef):
            start = _get_start_line(node)
            end = _get_end_line(node, total_lines)
            class_name = node.name

            # Extract source lines for this class
            chunk_content = "\n".join(lines[start - 1:end])
            docstring = ast.get_docstring(node)

            chunk_id = f"{rel_path}::{class_name}"
            chunks.append(CodeChunk(
                id=chunk_id,
                file_path=rel_path,
                language="python",
                symbol_name=class_name,
                symbol_type="class",
                start_line=start,
                end_line=end,
                content=chunk_content,
                imports=global_imports,
                docstring=docstring
            ))

            prev_class = self.current_class
            self.current_class = class_name
            # Visit methods inside class
            self.generic_visit(node)
            self.current_class = prev_class

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._handle_function(node, is_async=False)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._handle_function(node, is_async=True)

        def _handle_function(self, node: ast.AST, is_async: bool):
            start = _get_start_line(node)
            end = _get_end_line(node, total_lines)
            func_name = getattr(node, "name", "anonymous")
            docstring = ast.get_docstring(node)

            if self.current_class:
                symbol_type = "method"
                full_symbol = f"{self.current_class}.{func_name}"
            else:
                symbol_type = "function"
                full_symbol = func_name

            chunk_content = "\n".join(lines[start - 1:end])
            chunk_id = f"{rel_path}::{full_symbol}"

            chunks.append(CodeChunk(
                id=chunk_id,
                file_path=rel_path,
                language="python",
                symbol_name=full_symbol,
                symbol_type=symbol_type,
                start_line=start,
                end_line=end,
                content=chunk_content,
                imports=global_imports,
                docstring=docstring
            ))

            # Don't recurse into nested functions for separate top-level chunks if already included,
            # but visit to capture if desired. Visiting generic allows nested functions to also be chunked.
            self.generic_visit(node)

    visitor = CodeVisitor()
    visitor.visit(tree)

    # If the file has no classes or functions (e.g. script / config file), create a module chunk or line window
    if not chunks:
        if total_lines <= 100:
            chunks.append(CodeChunk(
                id=f"{rel_path}::module",
                file_path=rel_path,
                language="python",
                symbol_name=Path(rel_path).stem,
                symbol_type="module",
                start_line=1,
                end_line=total_lines,
                content=file_content,
                imports=global_imports,
                docstring=module_docstring
            ))
        else:
            chunks = _chunk_by_line_window(file_content, rel_path)

    return chunks


def chunk_file(file_path: Path, rel_path: str) -> List[CodeChunk]:
    """Read a file from disk and return its chunks."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return []
    return chunk_python_code(content, rel_path)

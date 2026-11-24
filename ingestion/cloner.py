import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import List, Generator, Tuple
from codesage.config import settings

class Cloner:
    """Clones remote git repositories or resolves local directory paths."""

    def __init__(self, cache_dir: Path = settings.repos_cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_repo_path(self, repo_url_or_path: str) -> Path:
        """Given a local path or a GitHub repository URL, return a local Path."""
        path_obj = Path(repo_url_or_path).expanduser().resolve()
        if path_obj.exists() and path_obj.is_dir():
            return path_obj

        # Assume it is a git URL
        return self.clone_repo(repo_url_or_path)

    def _get_repo_dirname(self, repo_url: str) -> str:
        parsed = urllib.parse.urlparse(repo_url)
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        # Replace slashes with underscores for a flat cache dir
        safe_name = path.replace("/", "_")
        return safe_name or "repo"

    def clone_repo(self, repo_url: str) -> Path:
        """Clone a git repository into cache_dir if not already cloned."""
        dirname = self._get_repo_dirname(repo_url)
        target_dir = self.cache_dir / dirname

        if target_dir.exists() and (target_dir / ".git").exists():
            return target_dir

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        # Clone using git CLI with depth 1 for speed
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository '{repo_url}': {result.stderr.strip()}")

        return target_dir


def walk_code_files(repo_path: Path) -> Generator[Tuple[Path, str], None, None]:
    """
    Walk directory and yield (absolute_file_path, relative_file_path).
    Filters out ignored directories, binary extensions, and oversized files.
    """
    repo_path = Path(repo_path).resolve()

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        # Prune ignored directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in settings.ignored_dirs and not d.startswith(".")
        ]

        for file_name in files:
            file_path = root_path / file_name
            ext = file_path.suffix.lower()

            if ext in settings.ignored_extensions:
                continue

            # Only accept supported code extensions for AST parsing (e.g. .py)
            if ext not in settings.supported_extensions:
                continue

            try:
                if file_path.stat().st_size > settings.max_file_size_bytes:
                    continue
            except OSError:
                continue

            rel_path = file_path.relative_to(repo_path).as_posix()
            yield file_path, rel_path

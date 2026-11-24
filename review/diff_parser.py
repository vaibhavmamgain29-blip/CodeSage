import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DiffHunk:
    old_start: int
    old_length: int
    new_start: int
    new_length: int
    header: str
    lines: List[str] = field(default_factory=list)
    annotated_lines: List[str] = field(default_factory=list)
    # Changed line numbers in new file
    added_lines: List[int] = field(default_factory=list)
    deleted_lines: List[int] = field(default_factory=list)

@dataclass
class FileDiff:
    old_path: Optional[str]
    new_path: Optional[str]
    is_new: bool = False
    is_deleted: bool = False
    hunks: List[DiffHunk] = field(default_factory=list)

    @property
    def target_path(self) -> str:
        return self.new_path or self.old_path or "unknown"

    @property
    def all_changed_lines(self) -> List[int]:
        changed = []
        for h in self.hunks:
            changed.extend(h.added_lines)
        return sorted(list(set(changed)))


HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)")

def parse_unified_diff(diff_text: str) -> List[FileDiff]:
    """Parse standard unified diff into FileDiff structures with line mappings."""
    files: List[FileDiff] = []
    current_file: Optional[FileDiff] = None
    current_hunk: Optional[DiffHunk] = None

    old_line_cursor = 0
    new_line_cursor = 0

    lines = diff_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if line.startswith("diff --git"):
            # New file diff block
            parts = line.split(" ")
            old_p = parts[2][2:] if len(parts) > 2 and parts[2].startswith("a/") else None
            new_p = parts[3][2:] if len(parts) > 3 and parts[3].startswith("b/") else None

            current_file = FileDiff(old_path=old_p, new_path=new_p)
            files.append(current_file)
            current_hunk = None
            i += 1
            continue

        if line.startswith("--- "):
            if current_file:
                raw_path = line[4:].strip()
                if raw_path == "/dev/null":
                    current_file.is_new = True
                    current_file.old_path = None
                elif raw_path.startswith("a/"):
                    current_file.old_path = raw_path[2:]
            i += 1
            continue

        if line.startswith("+++ "):
            if current_file:
                raw_path = line[4:].strip()
                if raw_path == "/dev/null":
                    current_file.is_deleted = True
                    current_file.new_path = None
                elif raw_path.startswith("b/"):
                    current_file.new_path = raw_path[2:]
            i += 1
            continue

        m = HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) else 1
            new_start = int(m.group(3))
            new_len = int(m.group(4)) if m.group(4) else 1
            hdr = line.strip()

            current_hunk = DiffHunk(
                old_start=old_start,
                old_length=old_len,
                new_start=new_start,
                new_length=new_len,
                header=hdr
            )
            old_line_cursor = old_start
            new_line_cursor = new_start

            if current_file is None:
                current_file = FileDiff(old_path="unknown", new_path="unknown")
                files.append(current_file)

            current_file.hunks.append(current_hunk)
            i += 1
            continue

        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk.lines.append(line)
                current_hunk.annotated_lines.append(f"L{new_line_cursor:<4d} | {line}")
                current_hunk.added_lines.append(new_line_cursor)
                new_line_cursor += 1
            elif line.startswith("-"):
                current_hunk.lines.append(line)
                current_hunk.annotated_lines.append(f"     | {line}")
                current_hunk.deleted_lines.append(old_line_cursor)
                old_line_cursor += 1
            elif line.startswith(" "):
                current_hunk.lines.append(line)
                current_hunk.annotated_lines.append(f"L{new_line_cursor:<4d} | {line}")
                old_line_cursor += 1
                new_line_cursor += 1
            elif line.startswith("\\ No newline at end of file"):
                current_hunk.lines.append(line)
                current_hunk.annotated_lines.append(f"     | {line}")
            else:
                # Outside hunk
                current_hunk = None

        i += 1

    return files

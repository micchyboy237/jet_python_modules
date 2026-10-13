"""
Agent Tools for repository exploration.
"""

from pathlib import Path
from typing import List

from jet_telemetry import tool
from utils import logger


class RepoTools:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    @tool(description="Read content of a specific file with optional line range.")
    def read_file(self, file_path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Read content of a specific file with optional line range."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return f"Error: File {file_path} not found."

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))

            content = "".join(lines[start_idx:end_idx])
            return f"Content of {file_path} (lines {start_line}-{end_idx if end_idx != len(lines) else 'end'}):\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool(
        description="Search for exact text patterns in the codebase. Use for finding specific function names or variables."
    )
    def grep_search(self, pattern: str, file_ext: str = "") -> List[str]:
        """Search for a pattern in the repository. Returns list of matching lines with file paths."""
        matches = []
        try:
            for file_path in self.repo_path.rglob("*"):
                if file_path.is_file() and (
                    not file_ext or file_path.suffix == file_ext
                ):
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            for i, line in enumerate(f, 1):
                                if pattern.lower() in line.lower():
                                    rel_path = file_path.relative_to(self.repo_path)
                                    matches.append(f"{rel_path}:{i}: {line.strip()}")
                    except:
                        continue
        except Exception as e:
            logger.error(f"Grep search error: {e}")

        return matches[:50]

    @tool(description="List contents of a directory.")
    def list_dir(self, dir_path: str = ".") -> str:
        """List contents of a directory."""
        try:
            target = self.repo_path / dir_path
            if not target.is_dir():
                return f"Error: {dir_path} is not a directory."

            items = []
            for item in sorted(target.iterdir()):
                prefix = "[DIR] " if item.is_dir() else "[FILE] "
                items.append(f"{prefix}{item.name}")

            return f"Contents of {dir_path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

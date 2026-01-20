# local_file_search_tool.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from smolagents import Tool


@dataclass
class LocalFileSearchTool(Tool):
    """
    Recursively searches for files in a local directory tree.
    Supports filename pattern matching and optional content search.
    """
    name = "local_file_search"
    description = (
        "Recursively search for files under a base directory. "
        "Can match filenames with glob patterns and/or search file contents for a substring."
    )
    inputs = {
        "base_dir": {
            "type": "string",
            "description": "Absolute or relative path to the root directory to search"
        },
        "pattern": {
            "type": "string",
            "description": 'Glob pattern, e.g. "*.py", "**/*.md", "*.txt"',
            "default": "**/*",
            "nullable": False
        },
        "content_contains": {
            "type": "string",
            "description": "Optional: only return files whose content contains this substring (case-insensitive)",
            "nullable": True
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 50)",
            "default": 50,
            "nullable": False
        }
    }
    output_type = "string"

    def forward(
        self,
        base_dir: str,
        pattern: str = "**/*",
        content_contains: Optional[str] = None,
        max_results: int = 50,
    ) -> str:
        base = Path(base_dir).expanduser().resolve()
        if not base.is_dir():
            return f"Error: '{base}' is not a directory or does not exist."

        matches: List[Path] = []
        search_text = content_contains.lower() if content_contains else None

        try:
            for path in base.rglob(pattern):
                if not path.is_file():
                    continue
                if search_text:
                    try:
                        text = path.read_text(encoding="utf-8", errors="ignore").lower()
                        if search_text not in text:
                            continue
                    except Exception:
                        continue  # skip unreadable files
                matches.append(path)
                if len(matches) >= max_results:
                    break
        except Exception as exc:
            return f"Error during search: {str(exc)}"

        if not matches:
            return f"No files found matching pattern '{pattern}' under {base}"

        lines = [f"Found {len(matches)} file(s):"]
        for p in matches:
            try:
                rel = p.relative_to(base)
            except ValueError:
                rel = p
            lines.append(f"  • {rel}  ({p.stat().st_size:,} bytes)")
        
        return "\n".join(lines)
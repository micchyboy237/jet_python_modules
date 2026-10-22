"""
Utility Functions
Common helpers for file I/O and path management.
"""

import json
from pathlib import Path
from typing import Any


def get_step_dir(step_index: int, output_dir: Path) -> Path:
    """Returns the directory path for a specific step."""
    step_dir = output_dir / "steps" / f"step_{step_index}"
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def save_json(path: Path, data: Any) -> str:
    """Save data as JSON and return rich-formatted link."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return _format_rich_link(path)


def save_text(path: Path, content: str) -> str:
    """Save text content to a file and return rich-formatted link."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return _format_rich_link(path)


def _format_rich_link(path: Path) -> str:
    """Format a path as a Rich console link."""
    # Use absolute path for the link target to ensure it works from any directory
    abs_path = path.resolve()
    # Format: [link=file:///absolute/path]filename[/link]
    return f"[link=file://{abs_path}]{path.name}[/link]"

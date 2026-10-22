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


def save_json(path: Path, data: Any):
    """Save data as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text(path: Path, content: str):
    """Save text content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

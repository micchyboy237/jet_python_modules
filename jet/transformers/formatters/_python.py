from __future__ import annotations

from pathlib import Path
from typing import Union

from black import (
    InvalidInput,
    Mode,
    TargetVersion,
    find_project_root,
    format_str,
)

StrOrPath = Union[str, Path]


def format_python(code_or_path: StrOrPath, /) -> str:
    """
    Format Python code provided either as a string or as a file path.

    - If `code_or_path` is a Path *and* the file exists, reads and formats the file.
    - If `code_or_path` is a Path and does NOT exist, treats its string value as code to format (not a path to a file).
    - If `code_or_path` is a Path and is a directory, raises IsADirectoryError.
    - Otherwise (if a string), treats it as Python code to format.

    Uses Black with default settings (line-length=88, modern Python targets).

    Args:
        code_or_path: Path to a Python file or a string (or Path acting as code) containing Python code

    Returns:
        Formatted Python code as string (with trailing newline preserved style)

    Raises:
        IsADirectoryError: When a Path points to a directory
        PermissionError: When file cannot be read due to permissions
        InvalidInput: When Black cannot parse the code (invalid syntax, etc.)
        OSError: For problems reading an actual file
    """
    if isinstance(code_or_path, Path):
        path = code_or_path
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise OSError(f"Failed to read file {path}: {exc}") from exc
        elif path.exists():
            raise IsADirectoryError(f"Expected a file, got directory: {path}")
        else:
            # Path does not exist → treat str(path) as Python code to format
            # (this allows passing Path("some/code/snippet.py") even if file missing)
            content = str(path)
    else:
        content = code_or_path

    try:
        root = find_project_root((Path.cwd(),))
        mode = Mode(
            target_versions={
                TargetVersion.PY39,
                TargetVersion.PY310,
                TargetVersion.PY311,
                TargetVersion.PY312,
            },
            line_length=88,
            string_normalization=True,
            is_pyi=False,
        )
        formatted = format_str(content, mode=mode)
    except InvalidInput as exc:
        msg = f"Cannot parse Python code (Black error):\n{exc}"
        if len(content) > 300:
            msg += "\n(input was too long to show)"
        raise InvalidInput(msg) from exc

    return formatted

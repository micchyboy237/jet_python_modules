from pathlib import Path

from smolagents import Tool


class LocalFileReadTool(Tool):
    """
    Read the full text content of a file under a specified base directory.
    """

    name = "local_file_read"
    description = (
        "Read the entire content of a file located under the specified base directory. "
        "Returns the raw content on success."
    )
    inputs = {
        "base_dir": {
            "type": "string",
            "description": "Absolute or relative path to the root directory (security boundary)",
        },
        "relative_path": {
            "type": "string",
            "description": "Relative path to the file from base_dir (do not include leading /)",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, base_dir: str, relative_path: str) -> str:
        base = Path(base_dir).expanduser().resolve()
        if not base.is_dir():
            return f"Error: '{base}' is not a valid directory"

        relative_path = str(Path(relative_path)).lstrip("/\\")
        target = (base / relative_path).resolve()

        if not str(target).startswith(str(base)):
            return "Error: Path traversal attempt detected - path escapes the base directory."

        if not target.is_file():
            return f"Error: File not found - {relative_path} (under {base})"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            return content
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {str(e)}"

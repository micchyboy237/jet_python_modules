from pathlib import Path

from smolagents import Tool


class LocalFileWriteTool(Tool):
    """
    Write text files under a configured work directory.
    """

    name = "local_file_write"
    description = "Write text content to a file under the configured work directory."

    inputs = {
        "relative_path": {
            "type": "string",
            "description": "Relative file path under the work directory",
        },
        "content": {
            "type": "string",
            "description": "Text content to write",
        },
    }

    output_type = "string"

    def __init__(self, work_dir: str):
        super().__init__()
        self.work_dir = work_dir

    def forward(self, relative_path: str, content: str) -> str:
        base = Path(self.work_dir).expanduser().resolve()
        target = (base / relative_path).resolve()

        # Prevent directory traversal
        if not str(target).startswith(str(base)):
            return "Error: path escapes work_dir"

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return f"Wrote file: {target.relative_to(base)}"

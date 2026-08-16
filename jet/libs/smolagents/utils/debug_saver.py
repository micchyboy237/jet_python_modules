# jet_python_modules/jet/libs/smolagents/utils/debug_saver.py
import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from rich.console import Console
from rich.text import Text


def get_next_call_number(base_dir: Path, prefix: str = "call_") -> int:
    if not base_dir.exists():
        return 1
    numbers = []
    for d in base_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                num = int(d.name[len(prefix) :])
                numbers.append(num)
            except ValueError:
                pass
    return max(numbers, default=0) + 1 if numbers else 1


class DebugSaver:
    """Handles all debug file writing – easy to mock/disable"""

    def __init__(
        self,
        tool_name: str,
        base_dir: Path | None = None,
        serializer: Callable[[str | dict], dict] | None = None,
        console: Console | None = None,
    ):
        self.tool_name = tool_name.lower().replace(" ", "_")
        self.base_dir = base_dir or (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / f"{self.tool_name}_logs"
        )
        self.serializer = serializer
        self.current_call_dir: Path | None = None
        self.console = console or Console()

    def _log_save(self, filepath: Path) -> None:
        """Log successful save with Rich console and resource link."""
        # Create relative path for display
        try:
            relative_path = filepath.relative_to(Path.cwd())
        except ValueError:
            relative_path = filepath

        # Create styled text with resource link
        text = Text()
        text.append("✓ Saved: ", style="bold green")

        # Add the path as a clickable resource link using the base name
        base_name = filepath.name
        text.append(base_name, style=f"link file://{filepath.absolute()}")

        # Optionally show the full path in dimmed style
        text.append(f"  ({relative_path})", style="dim")

        self.console.print(text)

    @contextmanager
    def new_call(self, request_data: dict | None = None):
        tool_base = self.base_dir
        tool_base.mkdir(parents=True, exist_ok=True)

        call_nr = get_next_call_number(tool_base)
        call_dir = tool_base / f"call_{call_nr:04d}"
        call_dir.mkdir(exist_ok=True)

        self.current_call_dir = call_dir

        if request_data is not None:
            self.save_json("request.json", request_data, indent=2)

        try:
            yield call_dir
        finally:
            self.current_call_dir = None

    def save(self, filename: str, content: str, encoding: str = "utf-8") -> None:
        if not self.current_call_dir:
            return
        path = self.current_call_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        self._log_save(path)

    def save_json(self, filename: str, obj, **json_kwargs):
        json_settings = {"indent": 2, "ensure_ascii": False, **json_kwargs}
        self.save(
            filename,
            json.dumps(obj, **json_settings, default=self.serializer),
        )

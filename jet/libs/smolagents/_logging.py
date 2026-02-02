import json
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path


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


@contextmanager
def structured_tool_logger(
    logs_dir: Path | None, tool_name: str, request_data: dict, verbose: bool = False
) -> Generator[tuple[Path, Callable[[str], None]], None, None]:
    if not logs_dir:
        yield Path(), lambda x: None
        return

    tool_dir = logs_dir / f"tool_{tool_name.lower().replace(' ', '_')}"
    tool_dir.mkdir(parents=True, exist_ok=True)

    call_nr = get_next_call_number(tool_dir)
    call_dir = tool_dir / f"call_{call_nr:04d}"
    call_dir.mkdir(exist_ok=True)

    (call_dir / "request.json").write_text(
        json.dumps(request_data, indent=2, ensure_ascii=False, default=str)
    )

    log_lines = []
    if verbose:
        log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {tool_name}")

    log_func = log_lines.append

    exc_type = exc_msg = exc_tb = None
    try:
        yield call_dir, log_func
        if verbose:
            log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH OK")
    except Exception as e:
        exc_type = type(e).__name__
        exc_msg = str(e)
        exc_tb = traceback.format_exc()
        if verbose:
            log_lines.append(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {exc_type}"
            )
        raise
    finally:
        if log_lines:
            (call_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
        if exc_type is not None:
            err_lines = [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {exc_type}",
                exc_msg,
                "",
                exc_tb.rstrip(),
            ]
            (call_dir / "error.txt").write_text("\n".join(err_lines) + "\n")

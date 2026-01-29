import json
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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
def tool_call_logger(
    logs_dir: Path | None,
    tool_name: str,
    request_data: dict | Any,
    verbose: bool = False,
):
    if not logs_dir:
        yield None
        return

    tool_dir = logs_dir / f"tool_{tool_name.lower().replace(' ', '_')}"
    tool_dir.mkdir(parents=True, exist_ok=True)

    call_nr = get_next_call_number(tool_dir)
    call_dir = tool_dir / f"call_{call_nr:04d}"
    call_dir.mkdir(exist_ok=True)

    # 1. Save request
    (call_dir / "request.json").write_text(
        json.dumps(request_data, indent=2, ensure_ascii=False, default=str)
    )

    log_lines = []
    error_lines = None

    if verbose:
        log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {tool_name}")

    try:
        yield (call_dir, log_lines.append)  # <- tool can append to log_lines

        # Success path
        if verbose:
            log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH OK")

    except Exception as exc:
        error_lines = [
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR",
            "",
            str(exc),
            "",
            traceback.format_exc().strip(),
        ]
        if verbose:
            log_lines.append(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAILED: {type(exc).__name__}"
            )

        raise  # re-raise so tool still fails normally

    finally:
        # Write log.txt always
        if log_lines:
            (call_dir / "log.txt").write_text("\n".join(log_lines) + "\n")

        # Write error.txt only if failed
        if error_lines:
            (call_dir / "error.txt").write_text("\n".join(error_lines) + "\n")


@contextmanager
def structured_tool_logger(
    logs_dir: Path | None, tool_name: str, request_data: dict, verbose: bool = False
) -> Generator[tuple[Path | None, Callable[[str], None]], None, None]:
    if not logs_dir:
        yield None, lambda x: None
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

    exc_info = None
    try:
        yield call_dir, log_func
        if verbose:
            log_lines.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FINISH OK")
    except Exception as e:
        exc_info = (type(e).__name__, str(e), traceback.format_exc())
        if verbose:
            log_lines.append(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {type(e).__name__}"
            )
        raise
    finally:
        if log_lines:
            (call_dir / "log.txt").write_text("\n".join(log_lines) + "\n")
        if exc_info:
            err_lines = [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {exc_info[0]}",
                exc_info[1],
                "",
                exc_info[2].rstrip(),
            ]
            (call_dir / "error.txt").write_text("\n".join(err_lines) + "\n")

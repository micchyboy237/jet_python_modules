from __future__ import annotations

import sys
from collections.abc import Sequence
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


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import difflib
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Simple Black formatter wrapper. Formats files in-place or prints formatted code.",
        epilog=(
            "Examples:\n"
            "  python formatters.py example.py               # format file in-place\n"
            "  python formatters.py 'def f(x):return x'      # format snippet and print\n"
            "  python formatters.py *.py --check             # check formatting in CI"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "targets",
        nargs="+",
        help="Python files to format (in-place) or code snippets to format and print",
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Don't write files back, just return status. Returns 0 if already formatted, 1 otherwise.",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show diff instead of overwriting (implies --check behavior)",
    )
    parser.add_argument(
        "-l",
        "--line-length",
        type=int,
        default=88,
        help="Line length to pass to Black (default: 88)",
    )

    args = parser.parse_args(argv)

    changed = False
    exit_code = 0

    for target in args.targets:
        p = Path(target)

        if p.is_file():
            # Existing file → format in-place (or check/diff)
            try:
                original = p.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Error reading {p}: {exc}", file=sys.stderr)
                exit_code = 1
                continue

            try:
                formatted = format_python(p)  # uses file path → reads again inside
            except Exception as exc:
                print(f"Cannot format {p}:\n{exc}", file=sys.stderr)
                exit_code = 1
                continue

            if original == formatted:
                print(f"{p} is already formatted.", file=sys.stderr)
                continue

            changed = True

            if args.check or args.diff:
                if args.diff:
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        formatted.splitlines(keepends=True),
                        fromfile=str(p),
                        tofile=f"{p} (formatted)",
                    )
                    sys.stdout.writelines(diff)
                else:
                    print(f"{p} would be reformatted", file=sys.stderr)
            else:
                # Actually write
                try:
                    p.write_text(formatted, encoding="utf-8")
                    print(f"Reformatted {p}", file=sys.stderr)
                except OSError as exc:
                    print(f"Failed to write {p}: {exc}", file=sys.stderr)
                    exit_code = 1

        else:
            # Treat as code snippet → always print to stdout
            try:
                formatted = format_python(target)
                print(formatted, end="")  # preserve trailing newline style
            except Exception as exc:
                print(f"Cannot format input:\n{exc}", file=sys.stderr)
                exit_code = 1

    if args.check and changed:
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

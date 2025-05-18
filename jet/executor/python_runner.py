import subprocess
import os
import re
from pathlib import Path
from typing import Optional, List, Tuple, Union
from tqdm import tqdm


def sort_key(path: str) -> Tuple[int, str]:
    filename = os.path.basename(path)
    match = re.match(r"(\d+)(.*)", filename)
    if match:
        num_part = int(match.group(1))
        rest = match.group(2)
        return (num_part, rest)
    return (float('inf'), filename)


def run_python_files_in_directory(
    target_dir: Union[str, Path],
    exclude_dirs: Optional[List[str]] = None,
    python_interpreter: str = "python",
    recursive: bool = False
) -> None:
    """
    Runs all Python files in the target directory and prints live logs.

    Args:
        target_dir (Path): The root directory to search for Python files.
        exclude_dirs (Optional[List[str]]): List of directory names to exclude.
        python_interpreter (str): Python executable to use. Default is 'python'.
        recursive (bool): Whether to search subdirectories recursively.
    """
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    exclude_dirs = set(exclude_dirs or [])

    if recursive:
        files = [
            f for f in target_dir.rglob("*.py")
            if not any(part in exclude_dirs for part in f.parts)
        ]
    else:
        files = [
            f for f in target_dir.glob("*.py")
            if not any(part in exclude_dirs for part in f.parts)
        ]

    files.sort(key=lambda f: sort_key(str(f.name)))

    print(
        f"\nRunning {len(files)} Python files in: {target_dir} (recursive={recursive})\n")

    for file_path in files:
        rel_path = file_path.relative_to(target_dir)
        print(f"\n▶️ Running: {rel_path}\n{'=' * 60}")

        process = subprocess.Popen(
            [python_interpreter, str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream logs line by line
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        process.wait()

        status = "✅ Success" if process.returncode == 0 else f"❌ Failed (code {process.returncode})"
        print(f"\n{'-' * 60}\n{status}: {rel_path}\n")

import subprocess
import os
import re
import json
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict
from tqdm import tqdm
from fnmatch import fnmatch
from datetime import datetime


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
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
    python_interpreter: str = "python",
    recursive: bool = False,
    output_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Runs all Python files in the target directory and prints live logs.
    Optionally writes execution results to a JSON file and individual log files.

    Args:
        target_dir (Union[str, Path]): The root directory to search for Python files.
        exclude_dirs (Optional[List[str]]): List of directory names to exclude.
        includes (Optional[List[str]]): List of filename patterns to include.
        excludes (Optional[List[str]]): List of filename patterns to exclude.
        python_interpreter (str): Python executable to use. Default is 'python'.
        recursive (bool): Whether to search subdirectories recursively.
        output_dir (Optional[Union[str, Path]]): Directory to save the JSON file and logs directory.
    """
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    # Initialize output directory and log paths
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        status_file = output_dir / "files_status.json"
        status_data: List[Dict[str, str]] = []
    else:
        logs_dir = None
        status_file = None
        status_data = []

    exclude_dirs = set(exclude_dirs or [])
    includes = includes or []
    excludes = excludes or []

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

    # Apply includes filter if specified
    if includes:
        files = [f for f in files if any(
            fnmatch(f.name, pattern) for pattern in includes)]

    # Apply excludes filter
    files = [f for f in files if not any(
        fnmatch(f.name, pattern) for pattern in excludes)]

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

        output_lines = []
        if process.stdout:
            for line in process.stdout:
                print(line, end="")
                output_lines.append(line)

        process.wait()

        status = "Success" if process.returncode == 0 else f"Failed (code {process.returncode})"
        print(
            f"\n{'-' * 60}\n{'✅' if process.returncode == 0 else '❌'} {status}: {rel_path}\n")

        # Prepare status data
        status_entry = {
            "file": str(rel_path),
            "status": status,
            "return_code": str(process.returncode),
            "timestamp": datetime.now().isoformat()
        }
        status_data.append(status_entry)

        # Write individual log file
        if logs_dir:
            log_file = logs_dir / f"{file_path.name}.log"
            with log_file.open('w') as f:
                f.write(f"Timestamp: {status_entry['timestamp']}\n")
                f.write(f"File: {rel_path}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Output:\n{''.join(output_lines)}\n")
                f.write("-" * 60 + "\n")

    # Write status JSON file
    if status_file:
        with status_file.open('w') as f:
            json.dump(status_data, f, indent=2)

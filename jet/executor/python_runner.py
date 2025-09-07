from typing import Literal
import subprocess
import os
import re
import json
from pathlib import Path
from typing import Literal, Union, Optional, List, Dict, Tuple
from tqdm import tqdm
from fnmatch import fnmatch
from datetime import datetime

from jet.file.utils import save_file
from jet.logger import CustomLogger
from jet.transformers.formatters import format_json


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
    output_dir: Optional[Union[str, Path]] = None,
    rerun_mode: Literal["all", "failed", "unrun", "failed_and_unrun"] = "all"
) -> None:
    """
    Runs Python files in the target directory based on the specified rerun mode.
    Preserves previous file statuses if not rerun.
    Args:
        target_dir (Union[str, Path]): The root directory to search for Python files.
        exclude_dirs (Optional[List[str]]): List of directory names to exclude.
        includes (Optional[List[str]]): List of filename patterns to include.
        excludes (Optional[List[str]]): List of filename patterns to exclude.
        python_interpreter (str): Python executable to use. Default is 'python'.
        recursive (bool): Whether to search subdirectories recursively.
        output_dir (Optional[Union[str, Path]]): Directory to save the JSON file and success/failed directories.
        rerun_mode (Literal["all", "failed", "unrun", "failed_and_unrun"]): Mode to determine which files to run.
            "all": Run all files.
            "failed": Run only files that previously failed.
            "unrun": Run only files that have not been run before.
            "failed_and_unrun": Run files that previously failed or have not been run.
    """
    logger = CustomLogger(name="")
    logger.debug(f"Input target_dir: {target_dir}, type: {type(target_dir)}")
    logger.debug(f"Input output_dir: {output_dir}, type: {type(output_dir)}")
    logger.debug(f"Input rerun_mode: {rerun_mode}")

    # Validate rerun_mode
    valid_modes = {"all", "failed", "unrun", "failed_and_unrun"}
    logger.debug(f"Checking rerun_mode: {rerun_mode}")
    if rerun_mode not in valid_modes:
        logger.error(
            f"Invalid rerun_mode: {rerun_mode}. Must be one of {valid_modes}")
        raise ValueError(
            f"Invalid rerun_mode: {rerun_mode}. Must be one of {valid_modes}")

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        logger.debug(f"Converted output_dir to Path: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        success_dir = output_dir / "success"
        failed_dir = output_dir / "failed"
        success_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)
        status_file = output_dir / "files_status.json"
        status_data: List[Dict[str, str]] = []
        if status_file.exists():
            with status_file.open('r') as f:
                logger.debug(f"Loading existing status file: {status_file}")
                status_data = json.load(f)
            logger.debug(f"Loaded status_data: {status_data}")
        else:
            logger.debug(
                f"No status file found at {status_file}, defaulting to run all files")
            rerun_mode = "all"  # Default to "all" if no status file exists
        main_log_file = output_dir / "main.log"
        logger = CustomLogger(str(main_log_file), name="", overwrite=True)
        logger.debug(f"Initialized logger with main_log_file: {main_log_file}")
    else:
        success_dir = None
        failed_dir = None
        status_file = None
        status_data = []
        logger.debug("No output_dir provided, using default logger")

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

    logger.debug(
        f"Found {len(files)} Python files before filtering: {[f.name for f in files]}")
    if includes:
        files = [f for f in files if any(
            fnmatch(f.name, pattern) for pattern in includes)]
    files = [f for f in files if not any(
        fnmatch(f.name, pattern) for pattern in excludes)]
    files.sort(key=lambda f: sort_key(str(f.name)))
    logger.debug(
        f"After include/exclude filtering: {len(files)} files: {[f.name for f in files]}")

    # Filter files based on rerun_mode
    existing_files = {entry["file"] for entry in status_data}
    logger.debug(f"Existing files from status: {existing_files}")
    if rerun_mode == "failed":
        failed_files = {
            entry["file"] for entry in status_data if entry["status"].startswith("Failed")}
        logger.debug(f"Failed files from status: {failed_files}")
        files = [f for f in files if str(
            f.relative_to(target_dir)) in failed_files]
    elif rerun_mode == "unrun":
        files = [f for f in files if str(
            f.relative_to(target_dir)) not in existing_files]
    elif rerun_mode == "failed_and_unrun":
        failed_files = {
            entry["file"] for entry in status_data if entry["status"].startswith("Failed")}
        logger.debug(f"Failed files for failed_and_unrun: {failed_files}")
        files = [f for f in files if str(f.relative_to(target_dir)) not in existing_files or
                 str(f.relative_to(target_dir)) in failed_files]
    logger.debug(
        f"After rerun_mode filtering ({rerun_mode}): {len(files)} files: {[str(f.relative_to(target_dir)) for f in files]}")

    logger.info(
        f"\nRunning {len(files)} Python files in: {target_dir} (recursive={recursive}, rerun_mode={rerun_mode})\n")

    # Update status_data with new runs, preserving existing entries
    new_status_data = [entry for entry in status_data if str(entry["file"]) not in
                       {str(f.relative_to(target_dir)) for f in files}]
    for file_path in files:
        rel_path = file_path.relative_to(target_dir)
        logger.debug(f"\n▶️ Running: {rel_path}\n{'=' * 60}")
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
                logger.teal(line, end="")
                output_lines.append(line)
        process.wait()
        status = "Success" if process.returncode == 0 else f"Failed (code {process.returncode})"
        logger.gray(
            f"\n{'-' * 60}\n{'✅' if process.returncode == 0 else '❌'} {status}: {rel_path}\n")
        status_entry = {
            "file": str(rel_path),
            "status": status,
            "return_code": str(process.returncode),
            "timestamp": datetime.now().isoformat()
        }
        new_status_data.append(status_entry)
        if success_dir and failed_dir:
            log_dir = success_dir if process.returncode == 0 else failed_dir
            relative_dir = file_path.parent.relative_to(target_dir)
            target_log_dir = log_dir / relative_dir
            target_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = target_log_dir / \
                f"{os.path.splitext(file_path.name)[0]}.log"
            logger = CustomLogger(str(log_file), overwrite=True)
            logger.orange(f"Logs: {log_file}")
            with log_file.open('w') as f:
                f.write(f"Timestamp: {status_entry['timestamp']}\n")
                f.write(f"File: {rel_path}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Output:\n{''.join(output_lines)}\n")
                f.write("-" * 60 + "\n")
    if status_file:
        logger.debug(f"Saving status file: {status_file}")
        save_file(new_status_data, status_file)

from typing import Literal
import subprocess
import os
import re
import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
from fnmatch import fnmatch
from datetime import datetime
from jet.file.utils import save_file
from jet.logger import CustomLogger


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
            "failed": Run only files that previously failed, or all files if none failed.
            "unrun": Run only files that have not been run before.
            "failed_and_unrun": Run files that previously failed or have not been run.
    """
    logger = CustomLogger(name="")
    logger.debug(f"Input target_dir: {target_dir}, type: {type(target_dir)}")
    logger.debug(f"Input output_dir: {output_dir}, type: {type(output_dir)}")
    logger.debug(f"Input rerun_mode: {rerun_mode}")

    valid_modes = {"all", "failed", "unrun", "failed_and_unrun"}
    if rerun_mode not in valid_modes:
        raise ValueError(f"Invalid rerun_mode: {rerun_mode}. Must be one of {valid_modes}")

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        success_dir = output_dir / "success"
        failed_dir = output_dir / "failed"
        success_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)

        status_file = output_dir / "files_status.json"
        status_data: List[Dict[str, str]] = []

        if status_file.exists():
            try:
                with status_file.open("r") as f:
                    status_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted status file: {status_file}, initializing empty data")
                status_data = []

        main_log_file = output_dir / "main.log"
        logger = CustomLogger(log_file=str(main_log_file), overwrite=True)
    else:
        success_dir = failed_dir = status_file = None
        status_data = []

    exclude_dirs = set(exclude_dirs or [])
    includes = includes or []
    excludes = excludes or []

    if recursive:
        files = [f for f in target_dir.rglob("*.py") if not any(part in exclude_dirs for part in f.parts)]
    else:
        files = [f for f in target_dir.glob("*.py") if not any(part in exclude_dirs for part in f.parts)]

    if includes:
        files = [f for f in files if any(fnmatch(f.name, pattern) for pattern in includes)]
    files = [f for f in files if not any(fnmatch(f.name, pattern) for pattern in excludes)]
    files.sort(key=lambda f: sort_key(str(f.name)))

    existing_files = {entry["file"] for entry in status_data}

    # Determine which files to rerun
    if rerun_mode == "failed":
        failed_files = {entry["file"] for entry in status_data if entry["status"].startswith("Failed")}
        if failed_files:
            files = [f for f in files if str(f.relative_to(target_dir)) in failed_files]
    elif rerun_mode == "unrun":
        files = [f for f in files if str(f.relative_to(target_dir)) not in existing_files]
    elif rerun_mode == "failed_and_unrun":
        failed_files = {entry["file"] for entry in status_data if entry["status"].startswith("Failed")}
        files = [
            f for f in files
            if str(f.relative_to(target_dir)) not in existing_files
            or str(f.relative_to(target_dir)) in failed_files
        ]

    logger.info(f"\nRunning {len(files)} Python files in {target_dir} (recursive={recursive}, rerun_mode={rerun_mode})\n")

    # Preserve order: start with previous data, remove ones to be rerun
    new_status_data = [
        entry for entry in status_data
        if str(entry["file"]) not in {str(f.relative_to(target_dir)) for f in files}
    ]

    # Run each file
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

        logger.gray(f"\n{'-' * 60}\n{'✅' if process.returncode == 0 else '❌'} {status}: {rel_path}\n")

        status_entry = {
            "file": str(rel_path),
            "status": status,
            "return_code": str(process.returncode),
            "timestamp": datetime.now().isoformat(),
        }

        # --- Update or replace existing status entry (preserve order) ---
        existing_idx = next(
            (i for i, e in enumerate(new_status_data) if e["file"] == str(rel_path)), None
        )
        if existing_idx is not None:
            new_status_data[existing_idx] = status_entry
        else:
            new_status_data.append(status_entry)

        # --- Write per-file log and remove old failed log on success ---
        if success_dir and failed_dir:
            relative_dir = file_path.parent.relative_to(target_dir)
            success_log_dir = success_dir / relative_dir
            failed_log_dir = failed_dir / relative_dir
            success_log_dir.mkdir(parents=True, exist_ok=True)
            failed_log_dir.mkdir(parents=True, exist_ok=True)

            log_filename = f"{os.path.splitext(file_path.name)[0]}.log"
            target_log_file = (
                success_log_dir / log_filename
                if process.returncode == 0
                else failed_log_dir / log_filename
            )

            # Remove old failed log if success
            if process.returncode == 0:
                old_failed_log = failed_log_dir / log_filename
                if old_failed_log.exists():
                    try:
                        old_failed_log.unlink()
                        logger.debug(f"Removed old failed log: {old_failed_log}")
                    except Exception as e:
                        logger.warning(f"Could not remove old failed log {old_failed_log}: {e}")

            # Save log file
            with target_log_file.open("w") as f:
                f.write(f"Timestamp: {status_entry['timestamp']}\n")
                f.write(f"File: {rel_path}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Output:\n{''.join(output_lines)}\n")
                f.write("-" * 60 + "\n")

            logger.orange(f"Logs: {target_log_file}")

    # --- Save updated status file ---
    if status_file:
        logger.debug(f"Saving status file: {status_file}")
        save_file(new_status_data, status_file)

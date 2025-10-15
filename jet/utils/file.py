from pathlib import Path
from typing import Dict, List, Union
import os
import fnmatch


def search_files(
    base_dir: str | list[str],
    extensions: list[str],
    include_files: list[str] = [],
    exclude_files: list[str] = [],
) -> list[str]:
    """
    Scrape directories for files matching specific extensions and inclusion/exclusion criteria.
    Args:
        base_dir (str | list[str]): Base directory or list of directories to search.
        extensions (list[str]): File extensions to include.
        include_files (list[str], optional): Files or directories to include (supports wildcards, partial, relative, or full paths). Defaults to [].
        exclude_files (list[str], optional): Files or directories to exclude (supports wildcards, partial, relative, or full paths). Defaults to [].
    Returns:
        list[str]: List of filtered file paths.
    """
    if isinstance(base_dir, str):
        base_dirs = [base_dir]
    else:
        base_dirs = base_dir
    files = []
    for dir_path in base_dirs:
        files.extend(
            os.path.join(root, f)
            for root, _, filenames in os.walk(dir_path)
            for f in filenames
            if any(f.endswith(ext) for ext in extensions)
        )

    if include_files:
        files = [file for file in files if matches_pattern(
            file, include_files)]
    if exclude_files:
        files = [file for file in files if not matches_pattern(
            file, exclude_files)]

    return sorted(files)  # Sort for consistent output in tests


def matches_pattern(path: str, patterns: list[str]) -> bool:
    """Check if the path, filename, or directories match any of the given patterns."""
    file_name = os.path.basename(path)
    dir_path = os.path.dirname(path)
    normalized_path = os.path.normpath(path)

    for pattern in patterns:
        # Normalize pattern for consistent matching
        pattern = os.path.normpath(pattern)

        # Full path match (relative or absolute) with wildcards
        if fnmatch.fnmatch(normalized_path, f"*{pattern}*") or fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Exact filename match
        if fnmatch.fnmatch(file_name, pattern):
            return True
        # Directory segments match (partial paths like "tutorial/agents")
        if any(fnmatch.fnmatch(part, pattern) or part == pattern for part in dir_path.split(os.sep)):
            return True
        # Handle relative path patterns
        if fnmatch.fnmatch(normalized_path, f"*{os.sep}{pattern}"):
            return True

    return False


def find_files_recursively(pattern: str, base_dir: Union[str, Path] = ".") -> List[str]:
    """
    Recursively find files matching the given pattern starting from base_dir.
    Args:
        pattern (str): Glob pattern (e.g., '*.py', '**/*.txt').
        base_dir (str | Path): Directory to start from. Defaults to current directory '.'.
    Returns:
        List[str]: List of matched file paths as strings.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise ValueError(f"Base path '{base_path}' is not a valid directory.")
    return [str(p) for p in base_path.rglob(pattern)]


def group_by_base_dir(paths: List[str], base_dir: str, max_depth: int = None) -> Dict[str, List[str]]:
    """
    Groups file paths by their shared base directory relative to the provided base_dir.
    Args:
        paths: List of file paths to group.
        base_dir: Base directory to compute relative paths from.
        max_depth: Maximum depth of directories to group by (optional).
    Returns:
        Dictionary mapping base directories to lists of file paths sharing that base.
    """
    base_path = Path(base_dir).resolve()
    grouped: Dict[str, List[str]] = {}

    for path in paths:
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = base_path / path
        try:
            full_path = full_path.resolve()
            relative = full_path.relative_to(base_path)
            parent_parts = relative.parent.parts
            # Apply max_depth limit if specified
            if max_depth is not None:
                parent_parts = parent_parts[:max_depth]
            parent = os.path.join(*parent_parts) if parent_parts else ""
            if parent not in grouped:
                grouped[parent] = []
            grouped[parent].append(str(full_path))
        except ValueError:
            continue

    return grouped

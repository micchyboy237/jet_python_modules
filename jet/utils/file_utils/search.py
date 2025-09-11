import glob
import os
from pathlib import Path
from typing import List, Set
import fnmatch


def find_files(
    base_dir: str,
    include: List[str],
    exclude: List[str],
    include_content_patterns: List[str],
    exclude_content_patterns: List[str],
    case_sensitive: bool = False,
    extensions: List[str] = []
) -> List[str]:
    normalized_extensions = [
        ext.lstrip('.').lstrip('*').lower() for ext in extensions
    ]
    matched_files: Set[str] = set()
    base_path = Path(base_dir).resolve()

    # Handle absolute paths in include
    for abs_path in [pat for pat in include if os.path.isabs(pat) and os.path.exists(pat)]:
        path = Path(abs_path)
        if path.is_file():
            if normalized_extensions:
                file_ext = path.suffix.lstrip('.').lower()
                if file_ext not in normalized_extensions:
                    continue
            normalized_path = os.path.normpath(
                str(path)).replace('/private/var', '/var')
            matched_files.add(normalized_path)
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    if normalized_extensions:
                        file_ext = file_path.suffix.lstrip('.').lower()
                        if file_ext not in normalized_extensions:
                            continue
                    normalized_path = os.path.normpath(
                        str(file_path)).replace('/private/var', '/var')
                    matched_files.add(normalized_path)

    # Adjust include and exclude patterns
    adjusted_include = []
    for pat in include:
        if not os.path.isabs(pat):
            if pat.endswith('/') or pat.endswith('/*/'):
                pat = pat.rstrip('/') + '/**/*'
            elif pat.startswith('*/'):
                pat = pat[2:].rstrip('/') + '/**/*'
            adjusted_include.append(pat)
        else:
            adjusted_include.append(pat)

    adjusted_exclude = []
    for pat in exclude:
        if not os.path.isabs(pat):
            if pat.endswith('/') or pat.endswith('/*'):
                pat = pat.rstrip('/') + '/**/*'
            adjusted_exclude.append(pat)
        else:
            adjusted_exclude.append(pat)

    # If no include patterns are provided, default to searching all files
    if not adjusted_include:
        adjusted_include = ['**/*']

    # Collect files based on include patterns
    for pattern in adjusted_include:
        if not os.path.isabs(pattern):
            for file_path in base_path.rglob(pattern):
                if file_path.is_file():
                    normalized_path = os.path.normpath(
                        str(file_path)).replace('/private/var', '/var')
                    matched_files.add(normalized_path)

    # Remove files based on exclude patterns
    files_to_remove = set()
    for pattern in adjusted_exclude:
        if not os.path.isabs(pattern):
            for file_path in base_path.rglob(pattern):
                if file_path.is_file():
                    normalized_path = os.path.normpath(
                        str(file_path)).replace('/private/var', '/var')
                    files_to_remove.add(normalized_path)
        else:
            exclude_path = Path(pattern)
            if exclude_path.is_file():
                normalized_path = os.path.normpath(
                    str(exclude_path)).replace('/private/var', '/var')
                files_to_remove.add(normalized_path)
            elif exclude_path.is_dir():
                for file_path in exclude_path.rglob('*'):
                    if file_path.is_file():
                        normalized_path = os.path.normpath(
                            str(file_path)).replace('/private/var', '/var')
                        files_to_remove.add(normalized_path)

    matched_files.difference_update(files_to_remove)

    # Apply extensions filter and content patterns
    final_files = []
    for file_path in matched_files:
        if os.path.isfile(file_path):
            if normalized_extensions:
                file_ext = Path(file_path).suffix.lstrip('.').lower()
                if file_ext not in normalized_extensions:
                    continue
            if matches_content(
                file_path, include_content_patterns, exclude_content_patterns, case_sensitive
            ):
                final_files.append(file_path)

    return sorted(final_files)


def matches_content(
    file_path: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    case_sensitive: bool = False
) -> bool:
    """
    Check if file content matches include patterns and does not match exclude patterns.

    Args:
        file_path (str): Path to the file to check.
        include_patterns (List[str]): Patterns to match in file content (e.g., "hello*").
        exclude_patterns (List[str]): Patterns to exclude from file content.
        case_sensitive (bool, optional): Whether matching is case-sensitive. Defaults to False.

    Returns:
        bool: True if content matches include patterns and does not match exclude patterns, False otherwise.

    Example:
        >>> matches_content("file.txt", ["hello"], ["world"], False)
        True
    """
    if not include_patterns and not exclude_patterns:
        return True
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content: str = f.read()
            if not case_sensitive:
                content = content.lower()
            if include_patterns:
                include_patterns = [
                    pattern if case_sensitive else pattern.lower() for pattern in include_patterns
                ]
                if not any(
                    fnmatch.fnmatch(
                        content, pattern) if '*' in pattern or '?' in pattern else pattern in content
                    for pattern in include_patterns
                ):
                    return False
            if exclude_patterns:
                exclude_patterns = [
                    pattern if case_sensitive else pattern.lower() for pattern in exclude_patterns
                ]
                if any(
                    fnmatch.fnmatch(
                        content, pattern) if '*' in pattern or '?' in pattern else pattern in content
                    for pattern in exclude_patterns
                ):
                    return False
        return True
    except (OSError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return False

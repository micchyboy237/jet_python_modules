import glob
import os
from typing import List, Set
import fnmatch

from jet.logger import logger


def find_files(
    base_dir: str,
    include: List[str],
    exclude: List[str],
    include_content_patterns: List[str],
    exclude_content_patterns: List[str],
    case_sensitive: bool = False,
    extensions: List[str] = []
) -> List[str]:
    """
    Find files in a directory matching include patterns, excluding specified patterns, and optionally filtering by content and extensions.
    Args:
        base_dir (str): The root directory to start the search.
        include (List[str]): Patterns or paths to include (e.g., "*.txt", "file.txt").
        exclude (List[str]): Patterns or paths to exclude (e.g., "*.py", "dir/*").
        include_content_patterns (List[str]): Patterns to match in file content (e.g., "hello*").
        exclude_content_patterns (List[str]): Patterns to exclude in file content.
        case_sensitive (bool, optional): Whether content matching is case-sensitive. Defaults to False.
        extensions (List[str], optional): File extensions to filter (e.g., "txt", ".txt", "*.txt"). Defaults to [].
    Returns:
        List[str]: List of relative file paths matching the criteria.
    Example:
        >>> find_files(
        ...     base_dir="/path",
        ...     include=["*.txt"],
        ...     exclude=["*.py"],
        ...     include_content_patterns=["hello"],
        ...     exclude_content_patterns=[],
        ...     case_sensitive=False,
        ...     extensions=["txt"]
        ... )
        ['file1.txt', 'dir/file2.txt']
    """
    # Normalize extensions to remove leading dots and wildcards
    normalized_extensions = [
        ext.lstrip('.').lstrip('*').lower() for ext in extensions
    ]

    logger.debug(f"Base Dir: {base_dir}")
    logger.debug(
        f"Finding files: include={include}, exclude={exclude}, extensions={extensions}")
    include_abs: List[str] = [
        os.path.relpath(path=pat, start=base_dir) if not os.path.isabs(
            pat) else pat
        for pat in include
        if os.path.exists(os.path.abspath(pat) if not os.path.isabs(pat) else pat)
    ]
    matched_files: Set[str] = set()
    adjusted_include: List[str] = [
        os.path.relpath(os.path.join(base_dir, pat), base_dir) if not any(
            c in pat for c in "*?") else pat
        for pat in include
    ]
    adjusted_exclude: List[str] = [
        os.path.relpath(os.path.join(base_dir, pat), base_dir) if not any(
            c in pat for c in "*?") else pat
        for pat in exclude
    ]
    candidate_files: Set[str] = set(include_abs)
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [
            d for d in dirs
            if not any(fnmatch.fnmatch(d, pat) or fnmatch.fnmatch(os.path.join(root, d), pat) for pat in adjusted_exclude)
        ]
        for file in files:
            file_path: str = os.path.relpath(
                os.path.join(root, file), base_dir)
            # Check if file extension matches any of the normalized extensions
            if normalized_extensions:
                file_ext = os.path.splitext(file)[1].lstrip('.').lower()
                if file_ext not in normalized_extensions:
                    continue
            is_wildcard_matched: bool = any(
                fnmatch.fnmatch(file_path, pat) for pat in include)
            absolute_include_matched: bool = any(
                fnmatch.fnmatch(os.path.abspath(file_path), os.path.abspath(pat)) for pat in include if "*" in pat
            )
            if (file_path in adjusted_include or is_wildcard_matched or absolute_include_matched) and not any(
                fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude
            ):
                if file_path not in candidate_files:
                    candidate_files.add(file_path)
                    logger.debug(
                        f"Candidate file (wildcard/include): {file_path}")
        for dir_name in dirs:
            dir_path: str = os.path.relpath(
                os.path.join(root, dir_name), base_dir)
            if any(fnmatch.fnmatch(dir_name, pat) or fnmatch.fnmatch(dir_path, pat) for pat in adjusted_include):
                for sub_root, _, sub_files in os.walk(os.path.join(root, dir_name).replace("*", "")):
                    base_sub_root: str = os.path.basename(sub_root)
                    if any(fnmatch.fnmatch(base_sub_root, pat) for pat in adjusted_exclude):
                        break
                    for file in sub_files:
                        file_path: str = os.path.relpath(
                            os.path.join(sub_root, file), base_dir)
                        # Check if file extension matches any of the normalized extensions
                        if normalized_extensions:
                            file_ext = os.path.splitext(
                                file)[1].lstrip('.').lower()
                            if file_ext not in normalized_extensions:
                                continue
                        if not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                            if file_path not in candidate_files:
                                candidate_files.add(file_path)
                                logger.debug(
                                    f"Candidate file (dir include): {file_path}")
        for file in files:
            file_path: str = os.path.relpath(
                os.path.join(root, file), base_dir)
            # Check if file extension matches any of the normalized extensions
            if normalized_extensions:
                file_ext = os.path.splitext(file)[1].lstrip('.').lower()
                if file_ext not in normalized_extensions:
                    continue
            is_current_package_json: bool = (
                file_path == "package.json" and "./package.json" in adjusted_include and root == base_dir
            )
            include_glob_matched: bool = any(
                path in file_path for include_path in include for path in glob.glob(include_path, recursive=True)
            )
            include_fnmatched: bool = any(fnmatch.fnmatch(
                file_path, pat) for pat in adjusted_include)
            exclude_fnmatched: bool = any(fnmatch.fnmatch(
                file_path, pat) for pat in adjusted_exclude)
            if (is_current_package_json or include_fnmatched or include_glob_matched) and not exclude_fnmatched:
                if file in adjusted_exclude:
                    continue
                if file_path not in candidate_files:
                    candidate_files.add(file_path)
                    logger.debug(
                        f"Candidate file (content match): {file_path}")
        include_dir_abs: List[str] = [
            pat for pat in include if "*" in pat and os.path.isdir(pat.replace("*", ""))
        ]
        for dir_name in include_dir_abs:
            dir_no_wildcard: str = dir_name.replace("*", "")
            dir_path: str = os.path.relpath(
                os.path.join(root, dir_name), base_dir)
            if any(fnmatch.fnmatch(dir_name, pat) or fnmatch.fnmatch(dir_path, pat) for pat in adjusted_include):
                for file in os.listdir(os.path.join(root, dir_no_wildcard)):
                    base_file: str = os.path.basename(file)
                    if any(fnmatch.fnmatch(base_file, pat) for pat in adjusted_exclude):
                        continue
                    file_path: str = os.path.relpath(
                        os.path.join(dir_no_wildcard, file), base_dir)
                    # Check if file extension matches any of the normalized extensions
                    if normalized_extensions:
                        file_ext = os.path.splitext(
                            file)[1].lstrip('.').lower()
                        if file_ext not in normalized_extensions:
                            continue
                    if not any(fnmatch.fnmatch(file_path, pat) for pat in adjusted_exclude):
                        rel_file_path: str = os.path.relpath(file_path)
                        if rel_file_path not in candidate_files and os.path.isfile(rel_file_path):
                            candidate_files.add(rel_file_path)
                            logger.debug(
                                f"Candidate file (include dir): {rel_file_path}")
    # Filter candidates by content patterns
    for file_path in candidate_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.isfile(full_path) and matches_content(
            full_path, include_content_patterns, exclude_content_patterns, case_sensitive
        ):
            matched_files.add(file_path)
            logger.debug(f"Added to final matched files: {file_path}")
    logger.debug(f"Final matched files: {matched_files}")
    return list(matched_files)


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

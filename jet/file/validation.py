import os
import fnmatch
import re
from jet.logger import logger


def match_placeholder_patterns(file_path: str, patterns: list[str]) -> bool:
    """Checks if a path matches any of the given patterns."""
    patterns_with_placeholder = [
        pattern for pattern in patterns if has_format_placeholders(pattern)]
    if not patterns_with_placeholder:
        return False
    for pattern in patterns_with_placeholder:
        placeholders = get_placeholders(pattern)
        for placeholder in placeholders:
            folder_path = os.path.join(
                file_path, pattern.replace("{"+placeholder+"}", "").lstrip("/"))
            if os.path.exists(folder_path):
                return True
    return False


def match_folder_patterns(file_path: str, patterns: list[str]) -> bool:
    """Checks if a path matches any of the given patterns."""
    patterns_with_placeholder = [
        pattern for pattern in patterns if has_folder_placeholders(pattern)]
    if not patterns_with_placeholder:
        return False
    for pattern in patterns_with_placeholder:
        folder_path = os.path.join(
            file_path, pattern.replace("<folder>", "").lstrip("/"))
        if os.path.exists(folder_path):
            return True
    return False


def match_double_wildcard_patterns(file_path: str, patterns: list[str]) -> bool:
    """Checks if a path matches any of the given patterns."""
    patterns_with_placeholder = [
        pattern for pattern in patterns if has_double_wildcard_placeholders(pattern)]
    if not patterns_with_placeholder:
        return False
    for pattern in patterns_with_placeholder:
        if not any([pattern.endswith("/*"), pattern.endswith("/**")]):
            final_pattern = pattern + "/**"
            return fnmatch.fnmatch(file_path, final_pattern)
    return False


def match_simple_folder_pattern(file_path: str, pattern: str) -> bool:
    """Checks if a path ends with a simple folder name pattern (e.g., '.venv')."""
    normalized_path = os.path.normpath(file_path)
    # Extract the base name of the path
    base_name = os.path.basename(normalized_path)
    # Normalize pattern by removing leading path separators, but keep '.' if it's part of the name
    normalized_pattern = os.path.basename(os.path.normpath(pattern))
    # Check if the base name matches the pattern exactly
    return base_name == normalized_pattern


def validate_match(file_path: str, include_patterns: list[str], exclude_patterns: list[str] = []) -> bool:
    paths = [file_path]
    included_by_placeholders = match_placeholder_patterns(
        file_path, include_patterns)
    included_by_folder_placeholders = match_folder_patterns(
        file_path, include_patterns)
    included_by_double_wildcards = match_double_wildcard_patterns(
        file_path, include_patterns)
    included_by_paths = any(
        fnmatch.fnmatch(path, pattern) for path in paths for pattern in include_patterns
    )
    # Check for simple folder name patterns (e.g., '.venv')
    included_by_simple_folder = any(
        match_simple_folder_pattern(file_path, pattern) for pattern in include_patterns
    )
    included = (included_by_placeholders or
                included_by_folder_placeholders or
                included_by_double_wildcards or
                included_by_paths or
                included_by_simple_folder)
    excluded_by_placeholders = match_placeholder_patterns(
        file_path, exclude_patterns)
    excluded_by_folder_placeholders = match_folder_patterns(
        file_path, exclude_patterns)
    excluded_by_double_wildcards = match_double_wildcard_patterns(
        file_path, exclude_patterns)
    excluded_by_paths = any(
        fnmatch.fnmatch(path, pattern) for path in paths for pattern in exclude_patterns
    )
    excluded = (excluded_by_placeholders or
                excluded_by_folder_placeholders or
                excluded_by_double_wildcards or
                excluded_by_paths)
    return included and not excluded


def match_format_placeholders(file_path: str, pattern: str) -> bool:
    normalized_path = os.path.normpath(file_path)
    normalized_pattern = os.path.normpath(pattern.lstrip('/'))
    return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")


def match_pattern(file_path: str, pattern: str) -> bool:
    normalized_path = os.path.normpath(file_path)
    normalized_pattern = os.path.normpath(pattern.lstrip('/'))
    return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")


def has_format_placeholders(text):
    return bool(re.search(r'(?<!\\)\{.*?\}', text))


def has_folder_placeholders(text):
    return "<folder>" in text


def has_double_wildcard_placeholders(text):
    return text.startswith("**/")


def get_placeholders(text):
    return re.findall(r'(?<!\\)\{(.*?)\}', text)


def format_with_placeholders(text, **kwargs):
    placeholders = get_placeholders(text)
    for placeholder in placeholders:
        if placeholder in kwargs:
            text = text.replace(f'{{{placeholder}}}', str(kwargs[placeholder]))
        else:
            raise KeyError(f"Missing value for placeholder: {placeholder}")
    return text

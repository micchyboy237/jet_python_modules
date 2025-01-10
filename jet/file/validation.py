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
    included = included_by_placeholders or included_by_folder_placeholders or included_by_double_wildcards or included_by_paths

    excluded_by_placeholders = match_placeholder_patterns(
        file_path, exclude_patterns)

    excluded_by_folder_placeholders = match_folder_patterns(
        file_path, exclude_patterns)

    excluded_by_double_wildcards = match_double_wildcard_patterns(
        file_path, exclude_patterns)
    excluded_by_paths = any(
        fnmatch.fnmatch(path, pattern) for path in paths for pattern in exclude_patterns
    )
    excluded = excluded_by_placeholders or excluded_by_folder_placeholders or excluded_by_double_wildcards or excluded_by_paths

    return included and not excluded


def match_format_placeholders(file_path: str, pattern: str) -> bool:
    """
    Matches a file path against a pattern that can include format placeholders {}
    """
    normalized_path = os.path.normpath(file_path)
    normalized_pattern = os.path.normpath(pattern.lstrip('/'))
    return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")


def match_pattern(file_path: str, pattern: str) -> bool:
    """
    Matches a file path against a pattern that can include folder components.
    """
    normalized_path = os.path.normpath(file_path)
    normalized_pattern = os.path.normpath(pattern.lstrip('/'))
    return fnmatch.fnmatch(normalized_path, f"*{normalized_pattern}")


def has_format_placeholders(text):
    # Matches non-escaped placeholders
    return bool(re.search(r'(?<!\\)\{.*?\}', text))


def has_folder_placeholders(text):
    return "<folder>" in text


def has_double_wildcard_placeholders(text):
    return text.startswith("**/")


def get_placeholders(text):
    """
    Extract all placeholders from the text.
    Returns a list of placeholders found within curly braces.
    """
    return re.findall(r'(?<!\\)\{(.*?)\}', text)


def format_with_placeholders(text, **kwargs):
    """
    Accepts text and kwargs for the placeholders.
    Replaces the placeholders in the text with values from kwargs.
    """
    placeholders = get_placeholders(text)
    for placeholder in placeholders:
        if placeholder in kwargs:
            text = text.replace(f'{{{placeholder}}}', str(kwargs[placeholder]))
        else:
            raise KeyError(f"Missing value for placeholder: {placeholder}")
    return text


def test_has_format_placeholders():
    # Test cases
    assert has_format_placeholders("Hello, {}!") == True, "Test 1 Failed"
    assert has_format_placeholders(
        "This is a test with escaped \\{placeholder\\}") == False, "Test 2 Failed"
    assert has_format_placeholders(
        "This is a \\{escaped placeholder\\} test.") == False, "Test 3 Failed"
    assert has_format_placeholders(
        "This is a \\{non-escaped\\} {placeholder}.") == True, "Test 4 Failed"


def test_get_placeholders():
    # Test cases
    text = "Hello, {name}! Welcome to {place}."

    # Test get_placeholders
    assert get_placeholders(text) == ['name', 'place'], "Test 1 Failed"


def test_format_placeholders():
    # Test cases
    text = "Hello, {name}! Welcome to {place}."

    # Test format_with_placeholders
    formatted_text = format_with_placeholders(text, name="John", place="Paris")
    assert formatted_text == "Hello, John! Welcome to Paris.", "Test 2 Failed"

    # Test missing placeholder
    try:
        format_with_placeholders(text, name="John")
        assert False, "Test 3 Failed (should raise KeyError)"
    except KeyError as e:
        assert "Missing value for placeholder: place" in str(
            e), "Test 3 Failed"


def test_validate_match():
    path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/.venv"

    logger.newline()
    logger.info(f"Path:")
    logger.debug(path)

    include_patterns = ["{base}/bin/activate"]
    exclude_patterns = []
    assert validate_match(path, include_patterns, exclude_patterns) == True

    include_patterns = ["<folder>/bin/activate"]
    exclude_patterns = []
    assert validate_match(path, include_patterns, exclude_patterns) == True

    include_patterns = ["**/.venv"]
    exclude_patterns = []
    assert validate_match(path, include_patterns, exclude_patterns) == True

    include_patterns = ["**/JetScripts"]
    exclude_patterns = []
    assert validate_match(path, include_patterns, exclude_patterns) == True

    include_patterns = ["**/JetScripts/*"]
    exclude_patterns = []
    assert validate_match(path, include_patterns, exclude_patterns) == True

    include_patterns = ["**/JetScripts/**"]
    exclude_patterns = ["**/test"]
    assert validate_match(path, include_patterns, exclude_patterns) == False

    include_patterns = ["**/JetScripts/*"]
    exclude_patterns = ["**/test/*"]
    assert validate_match(path, include_patterns, exclude_patterns) == False


# Example usage
if __name__ == "__main__":
    logger.newline()
    logger.orange("test_has_format_placeholders()...")
    test_has_format_placeholders()
    logger.success("All tests passed!")

    logger.newline()
    logger.orange("test_get_placeholders()...")
    test_get_placeholders()
    logger.success("All tests passed!")

    logger.newline()
    logger.orange("test_format_placeholders()...")
    test_format_placeholders()
    logger.success("All tests passed!")

    logger.newline()
    logger.orange("test_validate_match()...")
    test_validate_match()
    logger.success("All tests passed!")

import re

from bs4 import BeautifulSoup


def prettify_value(prompt, level=0):
    """
    Recursively builds a formatted log string from a nested dictionary or list.

    :param prompt: Dictionary or list to process.
    :param level: Indentation level for nested structures.
    :return: Formatted string for the log.
    """
    prompt_log = ""
    indent = " " * level  # Indentation for nested structures
    marker_list = ["-", "+"]
    marker = marker_list[level % 2]
    line_prefix = indent if level == 0 else f"{indent} {marker} "

    if isinstance(prompt, dict):
        for key, value in prompt.items():
            capitalized_key = key.capitalize()
            if isinstance(value, (dict, list)):  # If nested structure
                prompt_log += f"{line_prefix}{capitalized_key}:\n"
                prompt_log += prettify_value(value, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{capitalized_key}: {value}\n"
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, (dict, list)):  # If nested structure
                prompt_log += prettify_value(item, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{item}\n"

    return prompt_log


def minify_html(html: str) -> str:
    # Ensure whitespace between inline elements and surrounding text
    # Add space after closing inline tags if followed by text or another inline tag
    html = re.sub(r"(?<=[a-zA-Z0-9])(<(strong|em|span|a|b|i|code)[^>]*>)", r" \1", html)
    html = re.sub(r"(</(strong|em|span|a|b|i|code)>)(?=[a-zA-Z0-9<])", r"\1 ", html)

    # Explicitly add space between consecutive inline tags
    html = re.sub(
        r"(</(strong|em|span|a|b|i|code)>)\s*(<(strong|em|span|a|b|i|code)[^>]*>)",
        r"\1 \3",
        html,
    )

    # Reduce consecutive newlines and remove tabs
    html = re.sub(r"\s*\n\s*", "\n", html)
    return html


def format_html(
    html_string: str, parser: str = "html.parser", encoding: str | None = None
) -> str:
    """
    Beautify an HTML string by formatting it with proper indentation.

    Args:
        html_string (str): The input HTML string to beautify.
        parser (str, optional): The BeautifulSoup parser to use. Defaults to "html.parser".
        encoding (str, optional): The encoding for the output string. If None, returns a Unicode string.

    Returns:
        str: The beautified HTML string with proper indentation.

    Raises:
        TypeError: If html_string is not a string.
        ValueError: If the parser is invalid or html_string is empty.
    """
    if not isinstance(html_string, str):
        raise TypeError("html_string must be a string")
    if not html_string.strip():
        raise ValueError("html_string cannot be empty")

    try:
        soup = BeautifulSoup(html_string, parser)
        pretty_html = soup.prettify(encoding=encoding)
        # Handle byte string output when encoding is specified
        return (
            pretty_html.decode(encoding or "utf-8")
            if isinstance(pretty_html, bytes)
            else pretty_html
        )
    except Exception as e:
        raise ValueError(f"Failed to beautify HTML: {str(e)}")


# Example Usage
if __name__ == "__main__":
    # Example for prettify_value
    prompt = {
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {"email": "alice@example.com", "phone": "123-456-7890"},
        },
        "status": "active",
    }
    prompt_log = prettify_value(prompt)
    print("Formatted Prompt:")
    print(prompt_log)

    # Example for format_html
    html_input = "<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello <b>World</b></p></body></html>"
    formatted_html = format_html(html_input)
    print("\nFormatted HTML:")
    print(formatted_html)

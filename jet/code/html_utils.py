from typing import Optional
from bs4 import BeautifulSoup


def is_html(text: str) -> bool:
    """Check if the input string is valid HTML."""
    try:
        soup = BeautifulSoup(text, 'html.parser')
        return bool(soup.find())
    except:
        return False


def format_html(html_string: str, parser: str = "html.parser", encoding: Optional[str] = None) -> str:
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
        return pretty_html.decode(encoding or 'utf-8') if isinstance(pretty_html, bytes) else pretty_html
    except Exception as e:
        raise ValueError(f"Failed to beautify HTML: {str(e)}")

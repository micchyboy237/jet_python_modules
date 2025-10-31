from pathlib import Path
from typing import Union

from jet.code.html_utils import valid_html


def read_md_content(input: Union[str, Path], ignore_links: bool = False) -> str:
    """Read markdown content from a file path or string, optionally converting HTML to markdown."""
    from jet.code.markdown_utils import convert_html_to_markdown, clean_markdown_links
    md_content: str
    if isinstance(input, (str, Path)) and str(input).endswith(('.md', '.markdown', '.html', '.htm')) and Path(str(input)).is_file():
        with open(input, 'r', encoding='utf-8') as file:
            md_content = file.read()
        if str(input).endswith(('.html', '.htm')):
            md_content = convert_html_to_markdown(
                md_content, ignore_links=ignore_links)
    else:
        md_content = str(input)
        if valid_html(md_content):
            md_content = convert_html_to_markdown(
                md_content, ignore_links=ignore_links)

    if ignore_links:
        md_content = clean_markdown_links(md_content)

    return md_content


def read_html_content(input: Union[str, Path], ignore_links: bool = False) -> str:
    """
    Reads HTML content from a file path or string input, optionally removing links.

    Args:
        input: Path to the HTML file or a string containing HTML content.
        ignore_links: If True, links will be removed from the resulting markdown.

    Returns:
        The HTML content as a string.
    """
    html_content: str
    if isinstance(input, (str, Path)) and str(input).endswith(('.html', '.htm')) and Path(str(input)).is_file():
        with open(input, 'r', encoding='utf-8') as file:
            html_content = file.read()
    else:
        html_content = str(input)

    if not valid_html(html_content):
        raise ValueError(
            f"Invalid HTML content provided: {html_content[:100]}...")

    # # Optionally preprocess HTML (e.g., clean, normalize)
    # html_content = preprocess_html(html_content)

    # if ignore_links:
    #     md_content = convert_html_to_markdown(html_content, ignore_links=True)
    #     html_content = convert_markdown_to_html(md_content)
    #     return html_content

    return html_content

"""
HTML RAG Extractor using Trafilatura.
Extracts main content from HTML, cleans and formats it as markdown for RAG purposes.
Supports extraction with links and metadata preservation.
"""

from trafilatura import extract


def extract_content(
    html_content: str,
    url: str | None = None,
    include_links: bool = True,
    include_metadata: bool = True,
) -> str:
    """
    Extract main content from HTML using Trafilatura.

    Args:
        html_content: Raw HTML string to extract from
        url: Optional URL for better metadata extraction
        include_links: Whether to preserve hyperlinks in output (default: True)
        include_metadata: Whether to include metadata fields in output (default: True)

    Returns:
        Markdown-formatted extracted content with optional links and metadata

    Note:
        Trafilatura automatically strips boilerplate (navigation, ads, footers)
        while preserving the main article body and structure [[1]].
        The include_links parameter keeps links along with their targets [[1]].
        The with_metadata parameter extracts metadata fields like title, author, date [[2]].
    """
    result = extract(
        html_content,
        output_format="markdown",
        include_links=include_links,
        with_metadata=include_metadata,
        url=url,
    )
    return result if result else ""

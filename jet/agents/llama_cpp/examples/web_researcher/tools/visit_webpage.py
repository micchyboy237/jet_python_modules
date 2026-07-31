# web_researcher/tools/visit_webpage.py
"""
Visit webpage tool with content extraction and truncation.
"""

import re
from typing import Optional

import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(
    url: str,
    max_content_length: Optional[int] = 2000,
    extract_links: bool = False,
) -> str:
    """
    Visits a webpage and returns its content as truncated markdown text.

    Args:
        url: The URL of the webpage to visit.
        max_content_length: Maximum characters to return (default: 2000).
        extract_links: Whether to include links in output (default: False).

    Returns:
        The content of the webpage converted to Markdown, truncated to max length.
    """
    logger.info(f"Visiting webpage: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Convert HTML to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Extract links if requested
        if extract_links:
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown_content)
            if links:
                link_text = "\n\n**Links found:**\n"
                for text, link in links[:10]:
                    link_text += f"- [{text}]({link})\n"
                markdown_content += link_text

        # Truncate content
        if len(markdown_content) > max_content_length:
            markdown_content = (
                markdown_content[:max_content_length] + "\n... [content truncated]"
            )

        logger.info(f"Retrieved {len(markdown_content)} characters from {url}")
        return markdown_content

    except RequestException as e:
        error_msg = f"Error fetching webpage: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        return error_msg

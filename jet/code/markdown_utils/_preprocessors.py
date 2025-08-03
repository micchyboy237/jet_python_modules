import re
from typing import List

from jet.utils.text import fix_and_unidecode

from jet.logger import logger


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown text by removing unnecessary escape characters and trailing whitespace.

    Args:
        text: The markdown text to clean, or None.

    Returns:
        The cleaned text with unnecessary escapes removed and trailing whitespace trimmed, or None if input is None.
    """
    if text is None:
        return None
    # Remove escaped periods (e.g., "\." -> ".")
    text = re.sub(r'\\([.])', r'\1', text)
    # Trim trailing whitespace from each line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    text = fix_and_unidecode(text)
    return text


def clean_markdown_links(text: str) -> str:
    """
    Cleans markdown links in text, converting text links to their display text and removing image links.

    Args:
        text (str): Input markdown text, possibly multiline, containing text links [text](url) 
                   and/or image links ![alt](url).

    Returns:
        str: Text with markdown text links replaced by their display text and image links removed.
    """
    import re

    # Remove image links: ![alt](url) -> ''
    text = re.sub(r'!\[[^\]]*]\([^\)]*\)', '', text)

    # Replace text links: [text](url) or [text] (url) -> text or url if text is empty/whitespace
    def replace_link(match: re.Match[str]) -> str:
        link_text = match.group(1)
        url = match.group(2)
        return link_text.strip() if link_text.strip() else url

    # Match links with optional nested brackets
    text = re.sub(
        r'\[([^\[\]]*?(?:\[[^\[\]]*?\])*?[^\[\]]*?)\]\s*\(([^)]*?)\)',
        replace_link,
        text
    )

    # Preserve newlines and normalize spaces within lines
    parts = re.split(r'(\n+)', text)
    for i, part in enumerate(parts):
        if re.match(r'\n+', part):  # Skip newline parts
            continue
        if part.strip():  # Process non-empty parts after stripping
            # Preserve leading spaces, normalize internal spaces
            leading_match = re.match(r'^\s*', part)
            leading = leading_match.group(0) if leading_match else ''
            content = re.sub(r'[ \t]+', ' ', part.strip())
            parts[i] = leading + content
        else:  # Handle empty or whitespace-only parts
            parts[i] = ' ' if part else ''
    text = ''.join(parts)

    return text


def preprocess_markdown(md_content: str) -> str:
    """
    Preprocess markdown to normalize non-standard structures.

    Args:
        md_content: Raw markdown content.

    Returns:
        Normalized markdown content.
    """
    # Remove bold markdown syntax (e.g., **text** or __text__ to text)
    md_content = re.sub(r'\*\*(.*?)\*\*', r'\1', md_content)
    md_content = re.sub(r'__(.*?)__', r'\1', md_content)
    # Remove leading spaces from header lines
    md_content = re.sub(r'^\s*(#+)', r'\1', md_content, flags=re.MULTILINE)
    # Remove leading spaces from blockquote lines
    md_content = re.sub(r'^\s*(>+)', r'\1', md_content, flags=re.MULTILINE)
    # Ensure proper spacing around headers with links
    md_content = re.sub(r'^(#+)\s*\[([^\]]+)\]\(([^)]+)\)',
                        r'\1 [\2](\3)', md_content, flags=re.MULTILINE)

    # Remove bold markers but preserve headers
    md_content = re.sub(r'\*\*(.*?)\*\*', r'\1', md_content)
    md_content = re.sub(r'__(.*?)__', r'\1', md_content)
    # Fix incomplete regex for blockquotes
    md_content = re.sub(r'^\s*(>+)\s*', r'\1 ', md_content, flags=re.MULTILINE)
    # Fix task list regex to avoid affecting headers
    md_content = re.sub(r'^\s*([-*+])\s*\[([ xX])\]\s*(.*)',
                        r'\1 [\2] \3', md_content, flags=re.MULTILINE)
    # Remove consecutive spaces
    md_content = re.sub(r' +', ' ', md_content).strip()

    # Process separator lines
    md_content = process_separator_lines(md_content)

    md_content = clean_markdown_text(md_content)
    return md_content


def process_separator_lines(md_content: str) -> str:
    """
    Process markdown content to handle lines with only '-' or '*', moving next line's text to the right
    with a single space, or removing the separator if no text follows.

    Args:
        md_content: Raw markdown content as a string.

    Returns:
        Processed markdown content with transformed separator lines.
    """
    lines = md_content.splitlines()
    result: List[str] = []
    i = 0

    while i < len(lines):
        current_line = lines[i].strip()
        if current_line in ("-", "*"):
            # Check if there's a next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line:
                    # Combine separator and next line text
                    result.append(f"{current_line} {next_line}")
                    i += 2  # Skip the next line
                else:
                    # No text in next line, skip the separator
                    i += 1
            else:
                # Separator is the last line, skip it
                i += 1
        else:
            # Preserve non-separator lines
            result.append(lines[i])
            i += 1

    # Join lines, preserving original line endings
    return "\n".join(result)


__all__ = [
    "clean_markdown_text",
    "clean_markdown_links",
    "preprocess_markdown",
    "process_separator_lines",
]

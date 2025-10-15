import re
from typing import Dict, List, Optional, Set, Tuple, TypedDict
from urllib.parse import urljoin

from jet.utils.text import fix_and_unidecode



class MDHeaderLink(TypedDict):
    text: str
    url: str
    start_idx: int
    end_idx: int
    line: str
    line_idx: int
    is_heading: bool
    image_url: Optional[str]


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


def remove_markdown_links(text: str) -> str:
    """
    Remove all markdown links, replacing [label](link) with just label.
    """
    pattern = re.compile(r'\[([^\]]*)\]\((\S+?)(?:\s+"([^"]+)")?\)')
    output = ""
    last_end = 0

    for match in pattern.finditer(text):
        start, end = match.span()
        label = match.group(1)

        # Skip empty labels or labels with only spaces
        if not label.strip():
            output += text[last_end:start]
            last_end = end
            continue

        # Append text before this match
        output += text[last_end:start]

        # Replace with label only
        output += label

        last_end = end

    # Append remaining text after last match
    output += text[last_end:]

    return output


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


class LinkTextRatio(TypedDict):
    ratio: float
    is_link_heavy: bool
    link_chars: int
    total_chars: int
    cleaned_text_length: int


def link_to_text_ratio(text: str, threshold: float = 0.5) -> LinkTextRatio:
    """
    Calculates the ratio of link-related characters to total text characters in a markdown document.
    Returns whether the document is link-heavy based on a threshold.

    Args:
        text (str): Input markdown text containing possible links [text](url) or ![alt](url).
        threshold (float): Maximum link character proportion before flagging as link-heavy (default: 0.5); higher values loosen the check.

    Returns:
        dict: Contains the link-to-text ratio, whether it exceeds the threshold, and character counts.
              - ratio (float): Proportion of link-related characters to total characters.
              - is_link_heavy (bool): True if ratio is greater than or equal to threshold.
              - link_chars (int): Number of characters in links (including brackets, URLs, etc.).
              - total_chars (int): Total number of alphanumeric characters in the input text.
              - cleaned_text_length (int): Number of alphanumeric characters after removing links.
    """
    # Normalize input by removing leading/trailing whitespace and trailing punctuation
    text = text.strip().rstrip('.')

    # Get total alphanumeric characters
    total_chars = len(''.join(re.findall(r'[a-zA-Z0-9]', text)))

    # Clean the text to remove links and get the remaining content
    cleaned_text = clean_markdown_links(text)
    # Normalize cleaned text similarly
    cleaned_text = cleaned_text.strip().rstrip('.')
    cleaned_text_length = len(
        ''.join(re.findall(r'[a-zA-Z0-9]', cleaned_text)))

    # Calculate link characters (total - cleaned)
    link_chars = total_chars - cleaned_text_length

    # Calculate ratio (avoid division by zero)
    ratio = link_chars / total_chars if total_chars > 0 else 0.0

    # Determine if the document is link-heavy (include equality in threshold check)
    is_link_heavy = ratio >= threshold

    return {
        'ratio': ratio,
        'is_link_heavy': is_link_heavy,
        'link_chars': link_chars,
        'total_chars': total_chars,
        'cleaned_text_length': cleaned_text_length
    }


def extract_markdown_links(text: str, base_url: Optional[str] = None, ignore_links: bool = True) -> Tuple[List[MDHeaderLink], str]:
    """
    Extracts markdown links and plain URLs from text, optionally replacing them with their text content or cleaning URLs.
    Handles nested image links like [![alt](image_url)](link_url) and reference-style links like [text][ref].

    Args:
        text: Input string containing markdown links or plain URLs.
        base_url: Base URL to resolve relative links, if provided.
        ignore_links: If True, replaces links with text content in output; if False, preserves links.

    Returns:
        Tuple of a list of MDHeaderLink dictionaries and the modified text.
    """
    # Pattern for markdown links, including nested image links
    pattern = re.compile(
        r'\[(?:!\[([^\]]*?)\]\(([^)]+?)\)|([^\]]*))\]\((\S+?)\)|'
        # Capture reference-style links [text][ref]
        r'\[([^\]]*)\]\[([^\]]*)\]',
        re.MULTILINE
    )
    # Pattern for reference definitions [ref]: url
    ref_pattern = re.compile(
        r'^\[([^\]]*)\]:\s*(\S+)$',
        re.MULTILINE
    )
    plain_url_pattern = re.compile(
        r'(?<!\]\()https?://[^\s<>\]\)]+[^\s<>\]\).,?!]',
        re.MULTILINE
    )
    links: List[MDHeaderLink] = []
    output = text
    seen: Set[Tuple[str, str, str]] = set()  # Removed caption from key
    replacements: List[Tuple[int, int, str]] = []
    # Store reference URLs
    ref_urls: Dict[str, str] = {}  # Removed caption from ref_urls

    # Extract reference definitions first
    for match in ref_pattern.finditer(text):
        ref_id = match.group(1).strip()
        url = match.group(2).strip()
        if ref_id and url:
            ref_urls[ref_id.lower()] = url

    # Extract markdown links
    for match in pattern.finditer(text):
        start, end = match.span()
        image_alt, image_url, label, url, ref_text, ref_id = match.groups()
        selected_url = ""
        selected_image_url = image_url.strip() if image_url else None  # Capture image URL

        if ref_text and ref_id:  # Handle reference-style link [text][ref]
            label = ref_text
            if ref_id.lower() in ref_urls:
                selected_url = ref_urls[ref_id.lower()]
            else:
                continue  # Skip if reference not found
        else:
            label = image_alt if image_alt else label  # Use image alt as label if present
            # Prioritize outer link URL if present, otherwise use image URL
            selected_url = url.strip() if url else (image_url.strip() if image_url else "")

        if not selected_url:  # Skip if no valid URL
            continue

        # Convert relative URLs to absolute
        if base_url and not selected_url.startswith(('http://', 'https://')):
            selected_url = urljoin(base_url, selected_url)
        # Convert relative image URLs to absolute
        if base_url and selected_image_url and not selected_image_url.startswith(('http://', 'https://')):
            selected_image_url = urljoin(base_url, selected_image_url)

        # Find line and line index
        start_line_idx = text[:start].rfind('\n') + 1
        end_line_idx = text.find('\n', end)
        if end_line_idx == -1:
            end_line_idx = len(text)
        line = text[start_line_idx:end_line_idx].strip()
        line_idx = len(text[:start].splitlines()) - 1

        # Create link entry
        key = (label or "", selected_url, line)  # Key remains unchanged
        if key not in seen:
            seen.add(key)
            links.append({
                "text": label or "",
                "url": selected_url,
                "start_idx": start,
                "end_idx": end,
                "line": line,
                "line_idx": line_idx,
                "is_heading": line.startswith('#'),
                "image_url": selected_image_url
            })
        if ignore_links and label and label.strip():
            replacements.append((start, end, label))
        elif ignore_links:
            replacements.append((start, end, ""))
        else:
            replacements.append((start, end, match.group(0)))

    # Extract plain URLs (unchanged)
    for match in plain_url_pattern.finditer(text):
        url = match.group(0).strip()
        start, end = match.span()
        if not any(url in link["url"] for link in links):  # Avoid duplicates
            start_line_idx = text[:start].rfind('\n') + 1
            end_line_idx = text.find('\n', end)
            if end_line_idx == -1:
                end_line_idx = len(text)
            line = text[start_line_idx:end_line_idx].strip()
            line_idx = len(text[:start].splitlines()) - 1
            key = ("", url, None, line)
            if key not in seen:
                seen.add(key)
                links.append({
                    "text": "",
                    "url": url,
                    "start_idx": start,
                    "end_idx": end,
                    "line": line,
                    "line_idx": line_idx,
                    "is_heading": line.startswith('#'),
                    "image_url": None
                })
            if ignore_links:
                replacements.append((start, end, ""))
            else:
                replacements.append((start, end, url))

    # Apply replacements in reverse order
    if replacements:
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, replacement in replacements:
            output = output[:start] + replacement + output[end:]

    return links, output


__all__ = [
    "clean_markdown_text",
    "clean_markdown_links",
    "preprocess_markdown",
    "process_separator_lines",
    "link_to_text_ratio",
    "extract_markdown_links",
]

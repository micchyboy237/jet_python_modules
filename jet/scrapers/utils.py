import html
from dataclasses import dataclass
from datetime import datetime
from jet.code.html_utils import format_html
from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.data.header_docs import HeaderDocs
from jet.scrapers.config import TEXT_ELEMENTS
from jet.utils.url_utils import clean_url
from jet.wordnet.sentence import split_sentences
from lxml.etree import Comment
from typing import Callable, Optional, List, Dict, Set, TypedDict, Union
from bs4 import BeautifulSoup
import uuid
from jet.search.formatters import decode_text_with_unidecode
from jet.wordnet.words import count_words
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Tuple, TypedDict
from collections import defaultdict
import os
from typing import Generator, List, Optional
from urllib.parse import urljoin, urlparse
from jet.search.searxng import NoResultsFoundError, search_searxng, SearchResult
from pyquery import PyQuery as pq
from jet.logger.config import colorize_log
from jet.logger import logger
import json
import re
import string
from jet.utils.text import fix_and_unidecode
import parsel
import unidecode


# def scrape_links(html: str, base_url: Optional[str] = None) -> List[str]:
#     # Target attributes to extract
#     attributes = ['href', 'data-href', 'action']

#     # Build the pattern dynamically to support quoted values (single or double)
#     attr_pattern = '|'.join(attributes)
#     quote_pattern = (
#         rf'(?:{attr_pattern})\s*=\s*'      # attribute and equal sign
#         r'(["\'])'                         # opening quote (capture group 1)
#         r'(.*?)'                           # value (capture group 2)
#         r'\1'                              # matching closing quote
#     )

#     matches = re.findall(quote_pattern, html, flags=re.IGNORECASE)

#     # Define unwanted patterns
#     unwanted_extensions = (
#         r'\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|mp4|mp3|zip|rar|exe)$'
#     )
#     unwanted_paths = (
#         r'/(embed|api|static|assets|media|images|uploads)/'
#     )

#     # Filter and process links
#     filtered = []
#     for match in matches:
#         link = match[1].strip()

#         # Skip empty, javascript:, or mailto: links
#         if not link or link.lower().startswith(('javascript:', 'mailto:')):
#             continue

#         # Skip links with unwanted extensions
#         if re.search(unwanted_extensions, link, re.IGNORECASE):
#             continue

#         # Skip links with unwanted path patterns
#         if re.search(unwanted_paths, link, re.IGNORECASE):
#             continue

#         if base_url:
#             # Parse base_url to get scheme and netloc for filtering
#             parsed_base = urlparse(base_url)
#             base_scheme_netloc = f"{parsed_base.scheme}://{parsed_base.netloc}"

#             # Resolve relative URLs and anchor links
#             if link.startswith('#'):
#                 # Prepend base_url's scheme, netloc, and path to anchor links
#                 link = f"{base_scheme_netloc}{parsed_base.path}{link}"
#             else:
#                 # Resolve relative URLs against base_url
#                 link = urljoin(base_url, link)

#             # Filter links from the same domain (scheme and netloc)
#             if urlparse(link).netloc != parsed_base.netloc:
#                 continue
#         else:
#             # If no base_url, filter out fragment-only links and links without a host
#             if link == '#' or link.startswith('#'):
#                 continue
#             parsed_link = urlparse(link)
#             if not (parsed_link.scheme and parsed_link.netloc):
#                 continue

#         filtered.append(clean_url(link))

#     # Return unique links only
#     return list(dict.fromkeys(filtered))


def scrape_links(text: str, base_url: Optional[str] = None) -> List[str]:
    """
    Scrape all URLs from text, including absolute URLs and relative paths starting with '/'.
    If base_url is provided, convert relative paths to absolute URLs using base_url's scheme and host.

    Args:
        text: Input text to scrape for links
        base_url: Optional base URL to resolve relative paths

    Returns:
        List of unique URLs found in the text
    """
    # Regex pattern for URLs (absolute http(s) and relative paths starting with /)
    url_pattern = r'(?:(?:http|https)://[\w\-\./?:=&%#]+)|(?:/[\w\-\./?:=&%#]+[\w\-\./?:=&%#])'

    # Find all matches in text
    links = re.findall(url_pattern, text)

    if not links:
        return []

    # Handle base_url for relative paths
    if base_url:
        # Ensure base_url ends with a slash for proper joining
        if not base_url.endswith('/'):
            base_url = base_url + '/'

        # Convert relative paths to absolute URLs
        links = [
            urljoin(base_url, link) if link.startswith('/') else link
            for link in links
        ]

    # Remove duplicates while preserving order
    seen = set()
    unique_links = [
        link for link in links
        if not (link in seen or seen.add(link))
    ]

    # Validate URLs and filter out invalid ones
    valid_links = []
    parsed_base = urlparse(base_url) if base_url else None
    for link in unique_links:
        try:
            parsed = urlparse(link)
            # Only include http/https schemes with valid netloc
            if parsed.scheme in ('http', 'https') and parsed.netloc:
                # Exclude URLs that are just the base_url or base_url with slash
                if not (parsed_base and parsed.netloc == parsed_base.netloc and parsed.path in ('', '/')):
                    # Basic validation for path characters
                    if all(c not in '<>"\'' for c in link):
                        valid_links.append(link)
        except ValueError:
            continue

    return valid_links


class TitleMetadata(TypedDict):
    title: Optional[str]
    metadata: Dict[str, str]


def scrape_title(html: str) -> Optional[str]:
    """
    Scrape the title from an HTML string.

    Args:
        html: The HTML content to scrape.

    Returns:
        The page title as a string, or None if not found.
    """
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text().strip() if title_tag else None


def scrape_metadata(html: str) -> Dict[str, str]:
    """
    Scrape metadata from an HTML string.

    Args:
        html: The HTML content to scrape.

    Returns:
        A dictionary of metadata key-value pairs.
    """
    soup = BeautifulSoup(html, 'html.parser')
    metadata: Dict[str, str] = {}
    meta_tags = soup.find_all('meta')

    for meta in meta_tags:
        # Handle name or property attributes
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        if name and content:
            metadata[name] = content

        # Handle http-equiv meta tags
        http_equiv = meta.get('http-equiv')
        if http_equiv and content:
            metadata[f'http-equiv:{http_equiv}'] = content

        # Handle charset meta tags
        charset = meta.get('charset')
        if charset:
            metadata['charset'] = charset

    return metadata


def scrape_title_and_metadata(html: str) -> TitleMetadata:
    """
    Scrape the title and metadata from an HTML string.

    Args:
        html: The HTML content to scrape.

    Returns:
        A TitleMetadata dictionary containing the page title and metadata key-value pairs.
    """
    return {
        'title': scrape_title(html),
        'metadata': scrape_metadata(html)
    }


def scrape_published_date(html: str) -> str:
    """
    Scrape the published date from an HTML string and attempt to return it in ISO 8601 format.

    Args:
        html: The HTML content to scrape.

    Returns:
        The published date in ISO 8601 format (e.g., '2023-10-15T00:00:00Z'), or an empty string if not found or unparsable.
    """
    soup = BeautifulSoup(html, 'html.parser')
    meta_tags = soup.find_all('meta')

    date_keys = [
        'article:published_time',  # Open Graph
        'og:published_time',      # Open Graph alternative
        'dc.date',                # Dublin Core
        'dc.date.issued',         # Dublin Core
        'datePublished',          # Schema.org
        'pubdate',                # HTML5
        'publication_date',       # Generic
        'date'                    # Generic
    ]

    # Common date formats to try
    date_formats = [
        '%Y-%m-%dT%H:%M:%S%z',  # 2023-10-15T12:00:00Z or 2023-10-15T12:00:00+0000
        '%Y-%m-%d %H:%M:%S',   # 2023-10-15 12:00:00
        '%Y-%m-%d',            # 2023-10-15
        '%Y/%m/%d',            # 2023/10/15
    ]

    for meta in meta_tags:
        name = meta.get('name') or meta.get('property')
        content = meta.get('content')
        if name in date_keys and content:
            content = content.strip()
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(content, date_format)
                    return parsed_date.isoformat() + 'Z'  # Ensure UTC 'Z' suffix
                except ValueError:
                    continue

    # Check <time> tags
    time_tag = soup.find('time', pubdate=True) or soup.find(
        'time', datetime=True)
    if time_tag and time_tag.get('datetime'):
        content = time_tag.get('datetime').strip()
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(content, date_format)
                return parsed_date.isoformat() + 'Z'  # Ensure UTC 'Z' suffix
            except ValueError:
                continue

    return ""


def get_max_prompt_char_length(context_length: int, avg_chars_per_token: float = 4.0) -> int:
    """
    Calculate the maximum number of characters that can be added to a prompt.

    Parameters:
    - context_length (int): The context length in tokens.
    - avg_chars_per_token (float): Average number of characters per token. Default is 4.0.

    Returns:
    - int: Maximum number of characters for the prompt.
    """
    return int(context_length * avg_chars_per_token)


def clean_tags(root: parsel.Selector) -> parsel.Selector:
    """
    Remove style, script, navigation, images, and other non-text elements from the HTML.
    Remove anchor tags with hash links.
    Remove superscripts and subscripts.
    Retain only text-bearing elements.
    """
    # Exclude elements that don't contribute to visible text
    tags_to_exclude = ["style", "script", "nav", "footer",
                       "aside", "img", "sup", "sub"]
    for tag in tags_to_exclude:
        # Remove elements with the specified tag
        root.css(tag).remove()
    # Remove anchor tags with hash links
    root.css("a[href^='#']").remove()
    return root


def clean_text(text: str) -> str:
    """
    Clean the text by removing newlines, non-ASCII characters, and other characters.
    """
    # Convert Unicode characters to closest ASCII equivalent
    text = fix_and_unidecode(text)

    # text = ' '.join(lemmas).strip()
    text = clean_newlines(text)
    text = clean_markdown_formatting(text)
    # text = clean_spaces(text, exclude_chars=["-", "\n"])
    text = clean_non_ascii(text)
    text = clean_other_characters(text)

    return text.strip()


def clean_newlines(content, max_newlines: int = 2, strip_lines: bool = False) -> str:
    """
    Merge consecutive newlines from the content, but limit to at most max_newlines consecutive newlines.

    Args:
        content (str): The input text.
        max_newlines (int): Maximum allowed consecutive newlines.
        strip_lines (bool): If True, strip both leading and trailing whitespace from each line.

    Returns:
        str: The cleaned text.
    """
    if strip_lines:
        content = '\n'.join([line.strip() for line in content.split('\n')])
    else:
        content = '\n'.join([line.rstrip() for line in content.split('\n')])

    if max_newlines == 0:
        content = re.sub(r'\n+', ' ', content)
    else:
        content = re.sub(
            r'(\n{' + str(max_newlines + 1) + r',})', '\n' * max_newlines, content)

    return content


def clean_markdown_formatting(content: str) -> str:
    """
    Remove markdown formatting (bold, italic, strikethrough, code, headers, blockquotes, lists, etc.)
    while preserving the content text.

    Args:
        content (str): Input string with markdown formatting.

    Returns:
        str: String with markdown formatting removed.
    """

    # Remove bold (**text**, __text__)
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    content = re.sub(r'__(.*?)__', r'\1', content)

    # Remove italic (*text*, _text_)
    content = re.sub(r'\*(.*?)\*', r'\1', content)
    content = re.sub(r'_(.*?)_', r'\1', content)

    # Remove strikethrough (~~text~~)
    content = re.sub(r'~~(.*?)~~', r'\1', content)

    # Remove horizontal rules (---, ***, ___)
    content = re.sub(r'^\s*[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)

    return content.strip()


def clean_punctuations(content: str) -> str:
    """
    Replace consecutive and mixed punctuation marks (.?!), ensuring that each valid group
    is replaced with its last occurring punctuation, and replace all hyphens with spaces.

    Example:
        "Hello!!! How are you???" -> "Hello! How are you?"
        "Wait... What.!?" -> "Wait. What?"
        "Really...?!? Are you sure???" -> "Really. Are you sure?"
        "anime-strongest" -> "anime strongest"
        "data-test-123" -> "data test 123"
        "summer-2024" -> "summer 2024"

    Args:
        content: Input string with possible consecutive punctuations and hyphens.
    Returns:
        String with cleaned punctuation and hyphens replaced by spaces.
    """
    # Replace all hyphens with a space
    content = re.sub(r'-', ' ', content)
    # Replace consecutive punctuation with the last punctuation mark
    content = re.sub(r'([.?!]+)', lambda match: match.group()[-1], content)
    return content

# def clean_punctuations(content: str) -> str:
#     """
#     Replace consecutive and mixed punctuation marks (.?!), ensuring that each valid group
#     is replaced with its last occurring punctuation, and replace all punctuation between words with spaces.
#     Preserve decimal points in numbers (e.g., 3.14) and version numbers (e.g., 1.2.3) while treating standalone decimal points as punctuation.

#     Example:
#         "Hello!!! How are you???" -> "Hello! How are you?"
#         "Wait... What.!?" -> "Wait. What?"
#         "Really...?!? Are you sure???" -> "Really? Are you sure?"
#         "anime-strongest" -> "anime strongest"
#         "data.test.123" -> "data test 123"
#         "summer,2024" -> "summer 2024"
#         "Price: 3.14 dollars" -> "Price 3.14 dollars"
#         "Version...1.2.3" -> "Version.1.2.3"

#     Args:
#         content: Input string with possible consecutive punctuations and punctuation between words.
#     Returns:
#         String with cleaned punctuation and punctuation between words replaced by spaces.
#     """

#     # Preserve decimal points in numbers and version numbers
#     number_pattern = r'(\d+\.\d+(\.\d+)*)'
#     numbers = {}

#     def store_number(match):
#         key = f"__NUMBER_{len(numbers)}__"
#         numbers[key] = match.group(0)
#         return key

#     content = re.sub(number_pattern, store_number, content)

#     # Replace single punctuation between words with a space
#     content = re.sub(r'(?<=\w)[.,!?;:-](?=\w)', ' ', content)

#     # Replace consecutive punctuation with the last punctuation mark
#     content = re.sub(r'([.?!]+)', lambda match: match.group()[-1], content)

#     # Restore preserved numbers
#     for key, value in numbers.items():
#         content = content.replace(key, value)

#     return content


def protect_links(text: str) -> Tuple[str, List[str]]:
    """
    Protect markdown links and plain URLs by replacing them with placeholders.

    Args:
        text: Input string containing markdown links or plain URLs.

    Returns:
        Tuple containing the text with links replaced by placeholders and a list of the original links.
    """
    links = []
    replacements = []

    # Match markdown links: [text](url) or [text](url "caption")
    markdown_pattern = r'\[([^\]]*)\]\((\S+?)(?:\s+"([^"]+)")?\)'
    for match in re.finditer(markdown_pattern, text, re.MULTILINE):
        full_link = match.group(0)
        links.append(full_link)
        replacements.append((match.start(), match.end(), full_link))

    # Match plain URLs (http(s):// followed by non-whitespace characters, excluding markdown links)
    plain_url_pattern = r'(?<!\]\()https?://[^\s<>\]\)]+[^\s<>\]\).,?!]'
    for match in re.finditer(plain_url_pattern, text):
        full_url = match.group(0)
        if not any(full_url in link for link in links):
            links.append(full_url)
            replacements.append((match.start(), match.end(), full_url))

    # Sort replacements by start position to preserve text order
    replacements.sort(key=lambda x: x[0])

    # Replace links with unique placeholders in a single pass
    protected_text = text
    offset = 0
    for i, (start, end, link) in enumerate(replacements):
        placeholder = f"__LINK_{i}_{uuid.uuid4().hex[:8]}__"
        start += offset
        end += offset
        protected_text = protected_text[:start] + \
            placeholder + protected_text[end:]
        offset += len(placeholder) - (end - start)

    return protected_text, links


def restore_links(text: str, links: List[str]) -> str:
    """
    Restore protected links in the text by replacing placeholders with the original links.

    Args:
        text: Input string with placeholders.
        links: List of original links to restore.

    Returns:
        String with placeholders replaced by the original links.
    """
    restored_text = text
    for i, link in enumerate(links):
        placeholder = f"__LINK_{i}_[0-9a-f]{{8}}__"  # Match unique placeholder
        restored_text = re.sub(placeholder, link, restored_text)
    return restored_text


def clean_spaces(content: str) -> str:
    content, links = protect_links(content)

    # Remove spaces before .?!,;:])}
    content = re.sub(r'\s*([.?!,;:\]\)}])', r'\1', content)

    # Ensure single space *after* punctuation if followed by alphanum
    # content = re.sub(r'([.?!,;:\]\)}])(\w)', r'\1 \2', content)

    # Remove consecutive spaces
    content = re.sub(r' +', ' ', content).strip()

    # Remove empty brackets or brackets with only spaces
    content = re.sub(r'\[\s*\]', '', content)

    content = restore_links(content, links)
    return content


def clean_non_ascii(content: str) -> str:
    """Remove non-ASCII characters from the content."""
    return ''.join(i for i in content if ord(i) < 128)


def clean_other_characters(content: str) -> str:
    """Remove double backslashes from the content."""
    return content.replace("\\", "")


def clean_non_alphanumeric(text: str, include_chars: list[str] = []) -> str:
    """
    Removes all non-alphanumeric characters from the input string, except for optional included characters.

    :param text: The input string.
    :param include_chars: A list of additional characters to allow in the output.
    :return: A cleaned string with only alphanumeric characters and optional included characters.
    """
    if include_chars:
        allowed_chars = ''.join(re.escape(char) for char in include_chars)
        pattern = f"[^a-zA-Z0-9{allowed_chars}]"
    else:
        pattern = r"[^a-zA-Z0-9]"

    return re.sub(pattern, "", text)


def extract_sentences(content: str) -> list[str]:
    """Extract sentences from the content."""
    from jet.libs.txtai.pipeline import Textractor
    minlength = None
    textractor_sentences = Textractor(sentences=True, minlength=minlength)
    sentences = textractor_sentences(content)
    return sentences


def extract_paragraphs(content: str) -> list[str]:
    """Extract paragraphs from the content."""
    from jet.libs.txtai.pipeline import Textractor
    minlength = None
    textractor_paragraphs = Textractor(paragraphs=True, minlength=minlength)
    paragraphs = textractor_paragraphs(content)
    return paragraphs


def extract_sections(content: str) -> list[str]:
    """Extract sections from the content."""
    from jet.libs.txtai.pipeline import Textractor
    minlength = None
    textractor_sections = Textractor(sections=True, minlength=minlength)
    sections = textractor_sections(content)
    return sections


def merge_texts(texts: list[str], max_chars_text: int) -> list[str]:
    """Merge texts if it doesn't exceed the maximum number of characters."""
    merged_texts = []
    current_text = ""
    for text in texts:
        if len(current_text) + len(text) + 1 < max_chars_text:  # +1 for the newline character
            if current_text:
                current_text += "\n"  # Separate texts by newline
            current_text += text
        else:
            merged_texts.append(current_text)
            current_text = text
    if current_text:
        merged_texts.append(current_text)
    return merged_texts


def merge_texts_with_overlap(texts: List[str], max_chars_overlap: int = None) -> List[str]:
    merged_texts_with_overlaps = []

    for i in range(len(texts)):
        if i == 0:
            merged_texts_with_overlaps.append(texts[i])
        else:
            previous_text = texts[i - 1]
            current_text = texts[i]

            if not max_chars_overlap:
                merged_text = current_text
            else:
                overlap = previous_text[-max_chars_overlap:]
                merged_text = overlap + "\n" + current_text
            merged_texts_with_overlaps.append(merged_text)

    return merged_texts_with_overlaps


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def find_elements_with_text(html: str):
    """
    Finds all elements that contain any text, ensuring a natural document order.

    :param html: The HTML string to parse.
    :return: A list of dictionaries with parent elements and their corresponding text.
    """
    doc = pq(html)
    matching_parents = []
    seen_parents = set()

    for element in doc('*'):  # Iterate through all elements
        text = pq(element).text().strip()
        if text:  # If element contains text
            element_html = pq(element).outerHtml()
            if element_html not in seen_parents:  # Avoid duplicates
                matching_parents.append({
                    # PyQuery object (keeps all elements)
                    "parent": pq(element),
                    "text": text,           # The full text inside this element
                })
                seen_parents.add(element_html)  # Mark as seen

    return matching_parents  # Returns list of dictionaries


def extract_title_and_metadata(source: str, timeout_ms: int = 1000) -> TitleMetadata:
    """
    Extracts the <title> and relevant <meta> information from the HTML or dynamic content.

    :param source: HTML string or URL to parse.
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: Dictionary with title and metadata (meta[name]/[property] -> content).
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    # Use Playwright to render content dynamically if URL is provided or HTML is a complex structure
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract title and metadata
        title = page.title()
        metadata = {}

        for meta in page.query_selector_all("meta"):
            name = meta.get_attribute("name") or meta.get_attribute("property")
            content = meta.get_attribute("content")
            if name and content:
                metadata[name] = content

        browser.close()

    return {
        "title": title,
        "metadata": metadata
    }


def extract_internal_links(source: str, base_url: str, timeout_ms: int = 1000) -> List[str]:
    """
    Extracts all internal links from the HTML or dynamic content. These are links:
    - Starting with "/"
    - Or starting with the same domain as base_url

    :param source: HTML string or URL to parse.
    :param base_url: The base URL used to resolve and compare domains.
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: List of normalized internal links.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    internal_links = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract internal links from href, data-href, action attributes
        base_domain = urlparse(base_url).netloc
        url_attrs = ["href", "data-href", "action"]

        for elem in page.query_selector_all("*"):
            for attr in url_attrs:
                raw_url = elem.get_attribute(attr)
                if not raw_url:
                    continue
                raw_url = raw_url.strip()
                parsed = urlparse(raw_url)

                if raw_url.startswith("/"):
                    # Relative URL
                    full_url = urljoin(base_url, raw_url)
                    internal_links.add(full_url)
                elif parsed.netloc == base_domain:
                    # Absolute internal URL
                    internal_links.add(raw_url)

        browser.close()

    return sorted(internal_links)


def extract_clickable_texts_from_rendered_page(source: str, timeout_ms: int = 1000) -> List[str]:
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        # Wait for the document to be fully loaded
        page.wait_for_load_state("load")

        page.wait_for_timeout(timeout_ms)

        clickable_texts = page.evaluate("""
        () => {
            const elements = Array.from(document.querySelectorAll('*'));
            const clickable = [];

            for (const el of elements) {
                const isHidden = el.offsetParent === null || window.getComputedStyle(el).visibility === 'hidden';
                const isDisabled = el.disabled || el.getAttribute('aria-disabled') === 'true';

                // Check if the element is clickable based on its type
                const isClickableTag = ['A', 'BUTTON'].includes(el.tagName) ||
                    (el.tagName === 'INPUT' && ['button', 'submit'].includes(el.type.toLowerCase()));

                // Update the hasText condition to also check for 'value' on input[type="submit"]
                const hasText = (el.innerText || el.value || '').trim().length > 0;

                const hasClickAttr = el.hasAttribute('onclick');
                const hasClickHandler = (el.onclick !== null);

                const listeners = (window.getEventListeners && window.getEventListeners(el)?.click) || [];

                if ((isClickableTag || hasClickAttr || hasClickHandler || listeners.length > 0) &&
                    !isHidden && !isDisabled && hasText) {
                    clickable.push(el.innerText.trim() || el.value.trim());
                }
            }

            return clickable;
        }
        """)

        browser.close()

        if not clickable_texts:
            print("[warn] No clickable texts found.")
        return clickable_texts


def extract_element_screenshots(
    source: str,
    css_selectors: List[str] = [],
    output_dir: str = "generated/screenshots",
    timeout_ms: int = 1000
) -> List[str]:
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    screenshots = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
        )
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        for selector in css_selectors:
            try:
                elements = page.query_selector_all(selector)

                if len(elements) == 0:
                    print(f"[warn] No elements found for selector: {selector}")

                for i, el in enumerate(elements):
                    try:
                        el.wait_for_element_state("visible", timeout=1000)
                    except PlaywrightTimeoutError:
                        continue

                    clean_selector = selector.strip('.#> ').replace(
                        ' ', '_').replace('[', '').replace(']', '')
                    path = f"{output_dir}/{clean_selector}_{i}.png"
                    el.screenshot(path=path)
                    screenshots.append(path)

            except Exception as e:
                print(f"[warn] Failed to capture {selector}: {e}")

        browser.close()

    if not screenshots:
        print("[warn] No screenshots were taken.")

    return screenshots


def extract_form_elements(source: str, timeout_ms: int = 1000) -> List[str]:
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        form_elements = page.evaluate("""
        () => {
            const elements = Array.from(document.querySelectorAll('input, select, textarea, button, form'));
            const formElements = [];

            elements.forEach(el => {
                if (el.id) {
                    formElements.push('#' + el.id);
                }

                if (el.classList) {
                    el.classList.forEach(cls => {
                        if (!cls.startsWith('css-')) {
                            formElements.push('.' + cls);
                        }
                    });
                }
            });

            return formElements;
        }
        """)

        browser.close()

        if not form_elements:
            print("[warn] No form elements found.")
        return form_elements


def extract_search_inputs(source: str, timeout_ms: int = 1000) -> List[str]:
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        search_inputs = page.evaluate("""
        () => {
            const elements = Array.from(document.querySelectorAll('input[type="search"], input[type="text"]'));
            const searchElements = [];

            elements.forEach(el => {
                if (el.id) {
                    searchElements.push('#' + el.id);
                }

                if (el.classList) {
                    el.classList.forEach(cls => {
                        if (!cls.startsWith('css-')) {
                            searchElements.push('.' + cls);
                        }
                    });
                }
            });

            return searchElements;
        }
        """)

        browser.close()

        if not search_inputs:
            print("[warn] No search text inputs found.")
        return search_inputs


class BaseNode:
    """Base class for nodes with common attributes."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent_id: Optional[str] = None,
        class_names: List[str] = [],
        link: Optional[str] = None,
        line: int = 0,
        html: Optional[str] = None
    ):
        self.tag = tag
        self.text = text
        self.depth = depth
        self.id = id
        self.parent_id = parent_id
        self.class_names = class_names
        self.link = link
        self.line = line
        self.html = html
        self._parent_node: Optional['BaseNode'] = None

    def get_parent_node(self) -> Optional['BaseNode']:
        """
        Retrieves the parent node stored in the _parent_node attribute.

        Returns:
            The parent BaseNode if set, otherwise None.
        """
        return self._parent_node


class TreeNode(BaseNode):
    """A node representing an HTML element with its children."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent_id: Optional[str],
        class_names: List[str] = [],
        link: Optional[str] = None,
        children: Optional[List['TreeNode']] = None,
        line: int = 0,
        html: Optional[str] = None
    ):
        super().__init__(tag, text, depth, id, parent_id, class_names, link, line, html)
        self.children: List['TreeNode'] = children if children is not None else [
        ]

    def get_content(self) -> str:
        content = self.text or ""
        for child in self.children:
            content += child.get_content()
        return content.strip()

    def has_children(self) -> bool:
        """
        Checks if the node has any children.

        :return: True if the node has children, False otherwise.
        """
        return len(self.children) > 0

    def get_children(self) -> List['TreeNode']:
        """
        Returns the list of child nodes.

        :return: A list of TreeNode objects representing the children.
        """
        return self.children


def create_base_node(node: TreeNode) -> BaseNode:
    """
    Creates a BaseNode from a TreeNode, copying all relevant attributes.

    Args:
        node: The source TreeNode to copy attributes from.

    Returns:
        A new BaseNode instance with attributes copied from the TreeNode.
    """
    base_node = BaseNode(
        tag=node.tag,
        text=node.text,
        depth=node.depth,
        id=node.id,
        parent_id=node.parent_id,
        class_names=node.class_names,
        link=node.link,
        line=node.line,
        html=node.html
    )
    base_node._parent_node = node._parent_node
    return base_node


def exclude_elements(doc: pq, excludes: List[str]) -> None:
    """
    Removes elements from the document that match the tags in the excludes list.

    :param doc: The PyQuery object representing the HTML document.
    :param excludes: A list of tag names to exclude (e.g., ["style", "script"]).
    """
    for tag in excludes:
        for element in doc(tag):
            pq(element).remove()


def extract_tree_with_text(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 1000
) -> TreeNode:
    """
    Extracts a tree structure from HTML with id, parent_id, link attributes, actual line numbers, and outer HTML from formatted HTML.
    Sets the _parent_node attribute for each node if not already set.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)
        page_content = page.content()
        browser.close()

    # Format the HTML content using format_html
    formatted_html = format_html(page_content)
    doc = pq(formatted_html)
    exclude_elements(doc, excludes)

    # Split formatted HTML into lines for line number tracking
    html_lines = formatted_html.splitlines()

    root_el = doc[0]
    root_id = f"auto_{uuid.uuid4().hex[:8]}"
    tag_name = root_el.tag if isinstance(
        root_el.tag, str) else str(root_el.tag)

    # Use sourceline if available, otherwise default to 1
    root_line = getattr(root_el, 'sourceline', 1)

    # Set root node's html using outer_html
    root_node = TreeNode(
        tag=tag_name,
        text=None,
        depth=0,
        id=root_id,
        parent_id=None,
        class_names=[],
        link=None,
        children=[],
        line=root_line,
        html=pq(root_el).outer_html()
    )
    if root_node._parent_node is None:
        root_node._parent_node = None

    stack = [(root_el, root_node, 0)]  # (element, parent_node, depth)

    while stack:
        el, parent_node, depth = stack.pop()
        el_pq = pq(el)

        for child in el_pq.children():
            child_pq = pq(child)
            # Skip comment nodes
            if child.tag is Comment or str(child.tag).startswith('<cyfunction Comment'):
                continue

            tag = child.tag if isinstance(child.tag, str) else str(child.tag)
            class_names = [cls for cls in (child_pq.attr(
                "class") or "").split() if not cls.startswith("css-")]
            element_id = child_pq.attr("id")
            link = (
                child_pq.attr("href") or
                child_pq.attr("data-href") or
                child_pq.attr("action") or
                None
            )

            if not element_id or not re.match(r'^[a-zA-Z_-]+$', element_id):
                element_id = f"auto_{uuid.uuid4().hex[:8]}"

            # Handle text extraction based on tag type
            if tag.lower() == "meta":
                text = child_pq.attr("content") or ""
                text = decode_text_with_unidecode(text.strip())
            elif tag.lower() in TEXT_ELEMENTS:
                text_parts = []
                for item in child_pq.contents():
                    if isinstance(item, str):
                        text_parts.append(item.strip())
                    else:
                        text_parts.append(decode_text_with_unidecode(
                            pq(item).text().strip()))
                text = " ".join(part for part in text_parts if part)
            else:
                text = decode_text_with_unidecode(child_pq.text().strip())

            # Get the actual line number from lxml's sourceline if available
            line_number = getattr(child, 'sourceline', None)
            if line_number is None:
                # Fallback: Approximate line number by searching for the element's tag in the formatted HTML
                element_str = f"<{tag}"
                for i, line in enumerate(html_lines, 1):
                    if element_str in line:
                        line_number = i
                        break
                else:
                    line_number = parent_node.line + 1  # Fallback to parent line + 1 if not found

            child_node = TreeNode(
                tag=tag,
                text=text,
                depth=depth + 1,
                id=element_id,
                parent_id=parent_node.id,
                class_names=class_names,
                link=link,
                children=[],
                line=line_number,
                html=child_pq.outer_html()
            )
            if child_node._parent_node is None:
                child_node._parent_node = parent_node

            # Check if child text is a substring of parent text and empty parent text if true
            if parent_node.text and child_node.text and child_node.text in parent_node.text:
                parent_node.text = ""

            parent_node.children.append(child_node)

            # Only traverse deeper if the tag is not in TEXT_ELEMENTS
            if tag.lower() not in TEXT_ELEMENTS and child_pq.children():
                stack.append((child, child_node, depth + 1))

    return root_node


def flatten_tree_to_base_nodes(root: TreeNode) -> List[BaseNode]:
    """
    Flattens a TreeNode hierarchy into a list of BaseNode objects.

    Args:
        root: The root TreeNode to flatten.

    Returns:
        A list of BaseNode objects representing all nodes in the tree in depth-first order.
    """
    result: List[BaseNode] = []

    def traverse(node: TreeNode) -> None:
        # Create a BaseNode copy of the current node
        result.append(create_base_node(node))

        # Recursively process children
        for child in node.get_children():
            traverse(child)

    traverse(root)
    return result


def get_leaf_nodes(root: TreeNode) -> List[BaseNode]:
    """
    Returns a list of BaseNode objects for leaf nodes (nodes with no children).

    :param root: The root TreeNode of the tree to traverse.
    :return: A list of BaseNode objects for leaf nodes.
    """
    result: List[BaseNode] = []

    def traverse(node: TreeNode) -> None:
        # Check if the node is a leaf (no children)
        if not node.has_children():
            result.append(create_base_node(node))

        # Recursively traverse children
        for child in node.get_children():
            traverse(child)

    traverse(root)
    return result


def extract_by_heading_hierarchy(
    source: str,
    tags_to_split_on: List[Tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ],
    excludes: List[str] = ["nav", "footer", "script", "style"]
) -> List[TreeNode]:
    """
    Extracts a list of TreeNode hierarchies split by heading tags, avoiding duplicates,
    with heading text prepended by '#' based on header level.
    Sets the _parent_node attribute for each node if not already set.
    """
    results: List[TreeNode] = []
    parent_stack: List[Tuple[int, TreeNode]] = []
    seen_ids: Set[str] = set()

    def clone_node(node: TreeNode, new_parent_id: Optional[str] = None, new_depth: int = 0) -> TreeNode:
        """
        Clones a node with a new parent and depth, preserving structure without duplicating IDs.
        Sets _parent_node if not already set.
        """
        new_id = node.id if node.id not in seen_ids else f"auto_{uuid.uuid4().hex[:8]}"
        seen_ids.add(new_id)

        text = node.text
        if node.tag in [tag[1] for tag in tags_to_split_on]:
            for prefix, tag in tags_to_split_on:
                if tag == node.tag:
                    text = f"{prefix} {node.text.strip()}" if node.text else node.text
                    break

        cloned = TreeNode(
            tag=node.tag,
            text=text,
            depth=new_depth,
            id=new_id,
            parent_id=new_parent_id,
            class_names=node.class_names,
            link=node.link,
            children=[],
            line=node.line,
            html=node.html
        )
        if cloned._parent_node is None:
            cloned._parent_node = node._parent_node
        return cloned

    def traverse(node: TreeNode) -> None:
        nonlocal parent_stack, results

        if node.id in seen_ids:
            return

        if node.tag in [tag[1] for tag in tags_to_split_on]:
            level = next(i for i, t in enumerate(
                tags_to_split_on) if t[1] == node.tag)

            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_node = parent_stack[-1][1] if parent_stack else None
            depth = parent_node.depth + 1 if parent_node else 0

            heading_node = clone_node(
                node, parent_node.id if parent_node else None, depth)
            results.append(heading_node)
            seen_ids.add(heading_node.id)
            parent_stack.append((level, heading_node))

        else:
            if parent_stack:
                parent_node = parent_stack[-1][1]
                child_node = clone_node(
                    node, parent_node.id, parent_node.depth + 1)
                parent_node.children.append(child_node)
                seen_ids.add(child_node.id)

        for child in node.children:
            traverse(child)

    tree = extract_tree_with_text(source, excludes=excludes)
    if tree:
        traverse(tree)

    return results


class TextHierarchyResult(BaseNode):
    """A node representing a text hierarchy with combined content and links."""

    def __init__(
        self,
        tag: str,
        header: str,
        content: str,
        links: List[str],
        depth: int,
        id: str,
        parent_id: Optional[str] = None,
        parent_headers: List[str] = [],
        parent_header: Optional[str] = None,
        parent_content: Optional[str] = None,
        parent_level: Optional[int] = None,
        level: Optional[int] = None,
        line: int = 0,
        html: Optional[str] = None
    ):
        super().__init__(
            tag=tag,
            text=None,
            depth=depth,
            id=id,
            parent_id=parent_id,
            line=line,
            html=html
        )
        self.header = header
        self.content = content
        self.links = links
        self.parent_headers = parent_headers
        self.parent_header = parent_header
        self.parent_content = parent_content
        self.parent_level = parent_level
        self.level = level


def extract_texts_by_hierarchy(
    source: str,
    tags_to_split_on: List[Tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ],
    ignore_links: bool = True,
    excludes: List[str] = ["nav", "footer", "script", "style"],
) -> List[TextHierarchyResult]:
    """
    Extracts a list of TextHierarchyResult objects from HTML, each containing the tag, header, combined content of a heading
    and its descendants, a list of unique links, depth, id, parent_id, parent, parent_content, parent_level, level, line, parent_headers, and parent_node attributes.
    Filters out results without a header or content. Ensures parent_headers is ordered from root to immediate parent.
    """
    def get_header_level(header: str) -> int:
        """Get the header level of a markdown header or HTML header tag."""
        if header.startswith("#"):
            header_level = 0
            for c in header:
                if c == "#":
                    header_level += 1
                else:
                    break
            return header_level
        elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
            return int(header[1])
        else:
            raise ValueError(f"Invalid header format: {header}")

    def collect_text_and_links(node: TreeNode) -> Tuple[TextHierarchyResult, str, str]:
        texts = []
        htmls = []
        links = set()
        header = ""

        if node.tag in [tag[1] for tag in tags_to_split_on]:
            header = node.text.strip() if node.text else ""
        elif node.text and node.text.strip():
            if not (ignore_links and node.link):
                texts.append(node.text.strip())

        if node.link:
            links.add(node.link)

        if node.html:
            htmls.append(node.html)

        for child in node.children:
            child_result, child_text, _ = collect_text_and_links(child)
            if child_text:
                texts.append(child_text)
            links.update(child_result.links)
            if child_result.html:
                htmls.append(child_result.html)

        combined_content = "\n".join(text for text in texts if text)
        combined_html = "<div>\n" + \
            "\n".join(html for html in htmls if html) + "\n</div>"

        level = get_header_level(header) if header else None

        result = TextHierarchyResult(
            tag=node.tag,
            header=header,
            content=combined_content,
            links=list(links),
            depth=node.depth,
            id=node.id,
            parent_id=node.parent_id,
            parent_header=None,
            parent_content=None,
            parent_level=None,
            level=level,
            line=node.line,
            parent_headers=[],
            html=combined_html
        )
        result._parent_node = node._parent_node
        return result, combined_content, header

    heading_nodes = extract_by_heading_hierarchy(
        source, tags_to_split_on, excludes)

    # Collect full content, header, and level for each heading node
    id_to_content = {}
    id_to_header = {}
    id_to_level = {}
    id_to_node = {}
    header_stack: List[Tuple[str, int, str]] = []  # (header, level, id)
    results = []

    for node in heading_nodes:
        result, combined_content, header = collect_text_and_links(node)
        id_to_content[node.id] = combined_content
        id_to_header[node.id] = header
        id_to_level[node.id] = result.level
        id_to_node[node.id] = node
        results.append(result)

        # Update header stack for parent_headers
        if header and result.level is not None:
            # Pop headers with equal or higher (lower number) level
            while header_stack and header_stack[-1][1] >= result.level:
                header_stack.pop()
            # Add current header to stack
            header_stack.append((header, result.level, node.id))

    # Populate parent, parent_content, parent_level, and parent_headers
    for result in results:
        if result.parent_id:
            result.parent_header = id_to_header.get(result.parent_id, None)
            result.parent_content = id_to_content.get(result.parent_id, None)
            result.parent_level = id_to_level.get(result.parent_id, None)

        # Populate parent_headers using header_stack
        if result.header and result.level is not None:
            parent_headers = []
            for header, level, node_id in header_stack:
                if node_id == result.id:
                    continue  # Skip the current node's header
                # Only include headers with lower level (higher in hierarchy)
                if level < result.level:
                    parent_headers.append((header, level))
            # Sort by level (ascending) to ensure root-to-parent order
            parent_headers.sort(key=lambda x: x[1])
            result.parent_headers = [header for header, _ in parent_headers]

    # Filter out results without a header or content
    return [result for result in results if result.header and result.content]


def extract_text_elements(source: str, excludes: list[str] = ["nav", "footer", "script", "style"], timeout_ms: int = 1000) -> List[str]:
    """
    Extracts a flattened list of text elements from the HTML document, ignoring specific elements like <style> and <script>.
    Uses Playwright to render dynamic content if needed.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of text elements found in the HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    # Use Playwright to render the page if URL is provided
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract the content
        page_content = page.content()
        browser.close()

    # Parse the content with PyQuery after Playwright has rendered it
    doc = pq(page_content)

    # Apply the exclusion logic before extracting text
    exclude_elements(doc, excludes)

    def extract_text(element) -> List[str]:
        text = pq(element).text().strip()

        valid_id_pattern = r'^[a-zA-Z_-]+$'
        element_id = pq(element).attr('id')
        element_class = pq(element).attr('class')
        id = element_id if element_id and re.match(
            valid_id_pattern, element_id) else None
        class_names = [name for name in (element_class.split() if element_class else [])
                       if re.match(valid_id_pattern, name)]

        if text and len(pq(element).children()) == 0:
            return [text]

        text_elements = []
        for child in pq(element).children():
            text_elements.extend(extract_text(child))

        return text_elements

    # Start with the root element and gather all text elements in a flattened list
    text_elements = extract_text(doc[0])

    return text_elements


def extract_text_nodes(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 1000
) -> List[BaseNode]:
    """
    Extracts a list of BaseNode objects containing text from the HTML document, ignoring specific elements.
    Uses Playwright to render dynamic content if needed.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :return: A list of BaseNode objects representing text-containing elements in the HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"

    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source

    # Use Playwright to render the page if URL is provided
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if url:
            page.goto(url, wait_until="networkidle")
        else:
            page.set_content(html)

        page.wait_for_timeout(timeout_ms)

        # Extract the content
        page_content = page.content()
        browser.close()

    # Parse the content with PyQuery after Playwright has rendered it
    doc = pq(page_content)

    # Apply the exclusion logic before extracting nodes
    exclude_elements(doc, excludes)

    def extract_nodes(element, depth: int = 0, parent_id: Optional[str] = None) -> List[BaseNode]:
        nodes = []
        element_pq = pq(element)
        text = element_pq.text().strip()
        tag = element_pq[0].tag if element_pq else "unknown"

        valid_id_pattern = r'^[a-zA-Z_-]+$'
        element_id = element_pq.attr('id')
        element_class = element_pq.attr('class')
        id = element_id if element_id and re.match(
            valid_id_pattern, element_id) else f"node_{depth}_{len(nodes)}"
        class_names = [name for name in (element_class.split() if element_class else [])
                       if re.match(valid_id_pattern, name)]
        link = element_pq.attr('href') if tag == 'a' else None

        # Create a BaseNode if the element has text and no children
        if text and len(element_pq.children()) == 0:
            nodes.append(BaseNode(
                tag=tag,
                text=text,
                depth=depth,
                id=id,
                parent_id=parent_id,
                class_names=class_names,
                link=link,
                line=element_pq[0].sourceline if element_pq else 0
            ))

        # Recursively process children
        for child in element_pq.children():
            nodes.extend(extract_nodes(child, depth + 1, id))

        return nodes

    # Start with the root element and gather all text nodes
    text_nodes = extract_nodes(doc[0])

    return text_nodes

# Function to print the tree-like structure recursively


def print_html(html: str):
    tree = extract_tree_with_text(html)

    def print_tree(node: TreeNode, indent=0, excludes: list[str] = ["nav", "footer", "script", "style"]):
        if node:
            if node.tag in excludes:
                return

            has_text = bool(node.text)
            has_id = bool(node.id)
            has_class = bool(node.class_names)
            has_child_text = node.children and node.children[0].text

            if has_text or has_id or has_class or has_child_text:
                tag_text = node.tag
                if has_id:
                    tag_text += " " + colorize_log(f"#{node.id}", "YELLOW")
                if has_class:
                    tag_text += " " + colorize_log(
                        ', '.join(
                            [f".{class_name}" for class_name in node.class_names]), "ORANGE"
                    )

                if has_text:
                    logger.log(
                        ('  ' * indent + f"{node.depth}:"),
                        tag_text,
                        "-",
                        json.dumps(node.text[:30]),
                        colors=["INFO", "DEBUG", "GRAY", "SUCCESS"]
                    )
                else:
                    logger.log(
                        ('  ' * indent + f"{node.depth}:"),
                        tag_text,
                        colors=["INFO", "DEBUG"]
                    )

            for child in node.children:
                print_tree(child, indent + 1, excludes)

    return print_tree(tree)


def safe_path_from_url(url: str, output_dir: str) -> str:
    parsed = urlparse(url)

    # Sanitize host
    host = parsed.hostname or 'unknown_host'
    safe_host = re.sub(r'\W+', '_', host)

    # Last path segment without extension
    path_parts = [part for part in parsed.path.split('/') if part]
    last_path = path_parts[-1] if path_parts else 'root'
    last_path_no_ext = os.path.splitext(last_path)[0]

    # Build final safe path: output_dir_safe_host_last_path_no_ext
    return "_".join([output_dir, safe_host, last_path_no_ext])


def search_data(query: str, use_cache: bool = True, **kwargs) -> list[SearchResult]:
    filter_sites = []
    engines = [
        "google",
        "brave",
        "duckduckgo",
        "bing",
        "yahoo",
    ]

    try:
        results: list[SearchResult] = search_searxng(
            query_url="http://Jethros-MacBook-Air.local:3000/search",
            query=query,
            filter_sites=filter_sites,
            engines=engines,
            config={
                "port": 3101
            },
            use_cache=use_cache,
            **kwargs
        )
        if not results:
            raise NoResultsFoundError(f"No results found for query: '{query}'")
        return results
    except NoResultsFoundError as e:
        if use_cache:
            logger.warning(
                f"No results found for query: '{query}'. Recursively retrying with use_cache=False.")
            return search_data(query, use_cache=False, **kwargs)
        else:
            raise


async def scrape_urls(urls: list[str], *, max_depth: Optional[int] = 0, query: Optional[str] = None, **kwargs) -> AsyncGenerator[tuple[str, str], None]:
    from jet.scrapers.crawler.web_crawler import WebCrawler

    crawler = WebCrawler(urls=urls, max_depth=max_depth, query=query, **kwargs)
    try:
        async for result in crawler.crawl():
            yield result["url"], result["html"]
    finally:
        crawler.close()


def validate_headers(html: str, min_count: int = 5, min_avg_word_count: int = 20) -> bool:
    from jet.scrapers.preprocessor import html_to_markdown
    from jet.code.splitter_markdown_utils import count_md_header_contents

    md_text = html_to_markdown(html)
    header_count = count_md_header_contents(md_text)

    return header_count >= min_count

    # headers = get_md_header_contents(md_text)
    # header_texts = [header["content"] for header in headers]
    # header_word_counts = [count_words(text) for text in header_texts]
    # avg_word_count = sum(header_word_counts) / \
    #     len(header_word_counts) if header_word_counts else 0

    # return header_count >= min_count and avg_word_count >= min_avg_word_count


class SignificantNode(BaseNode):
    """A node representing a significant element with its outer HTML."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent: Optional[str] = None,
        class_names: List[str] = [],
        link: Optional[str] = None,
        line: int = 0,
        html: str = "",
        has_significant_descendants: bool = False
    ):
        super().__init__(tag, text, depth, id, parent, class_names, link, line)
        self.html = html  # Outer HTML of the node including all children
        # Indicates if the node has children
        self.has_significant_descendants = has_significant_descendants


def _node_to_outer_html(node: TreeNode) -> str:
    """
    Converts a TreeNode to its outer HTML representation, including all nested children.

    :param node: The TreeNode to convert.
    :return: The outer HTML string for the node and its descendants.
    """
    attributes = []
    if node.id:
        attributes.append(f'id="{node.id}"')
    if node.class_names:
        attributes.append(f'class="{" ".join(node.class_names)}"')

    attr_str = " ".join(attributes).strip()
    tag_open = f"<{node.tag.lower()}" + (f" {attr_str}" if attr_str else "")

    # For void elements that don't have closing tags
    void_elements = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"
    }

    if node.tag.lower() in void_elements:
        return f"{tag_open} />"

    # Include text and recursively process children
    content = node.text or ""
    for child in node.children:
        content += _node_to_outer_html(child)

    return f"{tag_open}>{content}</{node.tag.lower()}>"


def create_significant_node(
    node: TreeNode,
    parent_id: Optional[str],
    html: str,
    has_significant_descendants: bool
) -> SignificantNode:
    """
    Creates a SignificantNode from a TreeNode, copying all relevant attributes.

    Args:
        node: The source TreeNode to copy attributes from.
        parent_id: The ID of the nearest significant ancestor.
        html: The outer HTML of the node.
        has_significant_descendants: Whether the node has significant descendants.

    Returns:
        A new SignificantNode instance with attributes copied from the TreeNode.
    """
    significant_node = SignificantNode(
        tag=node.tag,
        text=node.text,
        depth=node.depth,
        id=node.id,
        parent=parent_id,
        class_names=node.class_names,
        link=node.link,
        line=node.line,
        html=html,
        has_significant_descendants=has_significant_descendants
    )
    significant_node._parent_node = node._parent_node
    return significant_node


def get_significant_nodes(root: TreeNode) -> List[SignificantNode]:
    """
    Returns a list of SignificantNode objects for nodes that either have a non-auto-generated ID
    or are one of the specified tags: footer, aside, header, main, nav, article, section.
    Each node includes its full outer HTML including all nested children and has_significant_descendants flag
    indicating if it has significant descendants. The parent field references the nearest significant ancestor,
    explicitly set to None for the root significant node. Nodes without an ID attribute are assigned a generated
    unique ID based on their tag and a counter to handle multiple instances.

    :param root: The root TreeNode of the tree to traverse.
    :return: A list of SignificantNode objects for matching nodes.
    """
    significant_tags = {"footer", "aside", "header",
                        "main", "nav", "article", "section"}
    result: List[SignificantNode] = []
    id_counter = {}  # Track counts for generated IDs per tag

    def traverse(node: TreeNode, significant_ancestor_id: Optional[str]) -> bool:
        # Check if node is significant (non-auto-generated ID or significant tag)
        is_significant = (node.id and not node.id.startswith(
            "auto_")) or node.tag.lower() in significant_tags

        # Determine the ID for the SignificantNode
        node_id = node.id
        if is_significant and (node.id is None or node.id == ""):
            # Generate a unique ID for nodes in significant_tags without an ID
            tag = node.tag.lower()
            id_counter[tag] = id_counter.get(tag, 0) + 1
            node_id = f"generated_{tag}_{id_counter[tag]}"

        # Set the significant ancestor ID for children: use node's ID if significant and non-auto-generated, else keep ancestor
        current_significant_ancestor_id = node_id if is_significant and node_id and not node_id.startswith(
            "auto_") else significant_ancestor_id

        # Check if any descendants are significant
        has_significant_descendants = False
        for child in node.children:
            if traverse(child, current_significant_ancestor_id):
                has_significant_descendants = True

        if is_significant:
            html = _node_to_outer_html(node)
            result.append(create_significant_node(
                node=node,
                parent_id=significant_ancestor_id,
                html=html,
                has_significant_descendants=has_significant_descendants
            ))

        # Return True if the node or any of its descendants are significant
        return is_significant or has_significant_descendants

    # Start with None to ensure root significant node has parent=None
    traverse(root, None)
    return result


class ParentWithSharedClass(TreeNode):
    """Represents a parent node with its shared class name and matching children, extending TreeNode."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent: Optional[str] = None,
        class_names: List[str] = [],
        link: Optional[str] = None,
        children: List[TreeNode] = [],
        line: int = 0,
        shared_class: str = "",
        matching_children_count: int = 0
    ):
        super().__init__(tag, text, depth, id, parent, class_names, link, children, line)
        self.shared_class = shared_class
        self.matching_children_count = matching_children_count


def get_parents_with_shared_class(root: TreeNode, class_name: Optional[str] = None) -> List[ParentWithSharedClass]:
    """
    Finds parent nodes with direct children sharing a specified or common class, excluding parents with children having different classes.

    :param root: The root TreeNode of the tree to search.
    :param class_name: Optional class name to match in child nodes. If None, finds parents with any shared class among children.
    :return: A list of ParentWithSharedClass objects sorted by number of matching children (descending).
    """
    result = []

    def traverse(node: TreeNode):
        if not node.has_children():
            return

        children = node.get_children()

        if class_name:
            # Count direct children with the specified class
            matching_children = [
                child for child in children if class_name in child.class_names]
            # Check if any child has a different class
            has_different_class = any(
                child.class_names and class_name not in child.class_names for child in children)
            if matching_children and not has_different_class:
                result.append(ParentWithSharedClass(
                    tag=node.tag,
                    text=node.text,
                    depth=node.depth,
                    id=node.id,
                    parent=node.parent,
                    class_names=node.class_names,
                    link=node.link,
                    line=node.line,
                    shared_class=class_name,
                    matching_children_count=len(matching_children),
                    children=matching_children
                ))
        else:
            # Collect all classes from direct children
            child_classes = {}
            for child in children:
                for cls in child.class_names:
                    child_classes[cls] = child_classes.get(cls, 0) + 1

            # Find classes shared by multiple children
            for cls, count in child_classes.items():
                if count > 1:
                    # Check if all children either have the shared class or no classes
                    has_different_class = any(
                        child.class_names and cls not in child.class_names
                        for child in children
                    )
                    if not has_different_class:
                        matching_children = [
                            child for child in children if cls in child.class_names]
                        result.append(ParentWithSharedClass(
                            tag=node.tag,
                            text=node.text,
                            depth=node.depth,
                            id=node.id,
                            parent=node.parent_id,
                            class_names=node.class_names,
                            link=node.link,
                            line=node.line,
                            shared_class=cls,
                            matching_children_count=len(matching_children),
                            children=matching_children
                        ))

        # Recursively check all children nodes
        for child in children:
            traverse(child)

    # Start traversal from root
    traverse(root)

    # Sort results by number of matching children (descending)
    result.sort(key=lambda x: x.matching_children_count, reverse=True)

    return result


__all__ = [
    "get_max_prompt_char_length",
    "clean_tags",
    "clean_text",
    "clean_spaces",
    "clean_newlines",
    "clean_non_ascii",
    "clean_other_characters",
    "extract_sentences",
    "extract_paragraphs",
    "extract_sections",
    "merge_texts",
    "merge_texts_with_overlap",
    "split_text",
    "find_elements_with_text",
    "extract_text_elements",
    "extract_tree_with_text",
    "print_html",
    "get_significant_nodes",
    "get_leaf_nodes",
    "get_parents_with_shared_class",
]


if __name__ == "__main__":
    # Example usage
    # model_max_chars = 32768
    # max_chars = get_max_prompt_char_length(model_max_chars)
    # print(f"Maximum characters for the prompt: {max_chars}")
    context_file = "generated/drivers_license/_main.md"
    with open(context_file, 'r') as f:
        context = f.read()

    # Extract sections from the content
    sections = extract_sections(context)
    print(sections)
    # Print lengths of sections
    print([len(section) for section in sections])

    # Merge sections if it doesn't exceed the maximum number of characters
    # Order should be maintained
    max_chars_chunks = 2000
    max_chars_overlap = 200
    merged_sections = merge_texts(sections, max_chars_chunks)
    merged_sections = merge_texts_with_overlap(sections, max_chars_overlap)
    print(merged_sections)
    # Print lengths of merged sections
    print([len(section) for section in merged_sections])

    # Get sections with the most and least number of characters
    sorted_sections = sorted(merged_sections, key=len)
    print(
        f"Least number of characters ({len(sorted_sections[0])} characters):\n{sorted_sections[0]}")
    print(
        f"Most number of characters ({len(sorted_sections[-1])} characters):\n{sorted_sections[-1]}")

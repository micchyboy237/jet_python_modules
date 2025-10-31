import requests
import validators
import uuid
import os
import json
import re
import parsel
import base64

from datetime import datetime
from lxml.etree import Comment
from typing import Literal, Optional, List, Dict, Set, TypedDict, Union
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pathlib import Path
from typing import AsyncGenerator, Tuple
from urllib.parse import urljoin, urlparse
from pyquery import PyQuery as pq
from fake_useragent import UserAgent

from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from jet.transformers.formatters import format_html
from jet.scrapers.config import TEXT_ELEMENTS
from jet.search.formatters import decode_text_with_unidecode
from jet.search.searxng import NoResultsFoundError, search_searxng, SearchResult
from jet.logger.config import colorize_log
from jet.logger import logger
from jet.utils.text import fix_and_unidecode
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name


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
    url_pattern = r'(?:(?:http|https)://[\w\-\./?:=&%#]+)|(?:/[\w\-\./?:=&%#]*)'

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
            # Include relative paths starting with '/' even without base_url
            if parsed.path.startswith('/') and not parsed.scheme and not base_url:
                valid_links.append(link)
                continue
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


def extract_favicon_ico_link(source: str) -> Optional[str]:
    """
    Extracts the favicon.ico link from a given URL or HTML string.
    
    Args:
        source: The URL of the website or HTML string to extract the favicon from.
        
    Returns:
        The absolute URL of the favicon.ico if found, None otherwise.
    """
    try:
        # Check if source is a valid URL
        if validators.url(source):
            # Send HTTP request with a timeout
            response = requests.get(source, timeout=5)
            response.raise_for_status()
            html_content = response.text
            base_url = source
        else:
            # Treat source as HTML string
            html_content = source
            # Use a default base URL for relative paths in HTML string
            base_url = "https://example.com"

        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for favicon in <link> tags
        favicon_link = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        
        if favicon_link and favicon_link.get('href'):
            # Convert relative URL to absolute
            return urljoin(base_url, favicon_link['href'])
        
        # Fallback: try default /favicon.ico path (only for valid URLs)
        if validators.url(source):
            parsed_url = urlparse(source)
            default_favicon = f"{parsed_url.scheme}://{parsed_url.netloc}/favicon.ico"
            
            # Verify if default favicon exists
            favicon_response = requests.head(default_favicon, timeout=5)
            if favicon_response.status_code == 200:
                return default_favicon
            
        return None
        
    except (requests.RequestException, ValueError):
        return None


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


def get_xpath(element) -> str:
    # Helper to build the xpath string for an element in the PyQuery tree
    path_elems = []
    # element is a lxml Element inside PyQuery
    current = element
    while current is not None and hasattr(current, "tag"):  # root's parent is None
        parent = current.getparent()
        tag = current.tag

        # Find the index (nth position among same tag siblings)
        if parent is not None:
            same_tag_siblings = [sib for sib in parent if sib.tag == tag]
            if len(same_tag_siblings) == 1:
                idx = ""
            else:
                # XPath is 1-based
                idx = "[%d]" % (same_tag_siblings.index(current) + 1)
            path_elems.append(f"{tag}{idx}")
        else:
            path_elems.append(f"{tag}")
        current = parent
    xpath = "/" + "/".join(reversed(path_elems))
    return xpath


class BaseNode:
    """Base class for nodes with common attributes."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        raw_depth: Optional[int] = None,  # Added to store unadjusted DOM depth
        class_names: List[str] = [],
        line: int = 0,
        xpath: Optional[str] = None,
        html: Optional[str] = None,
    ):
        self.tag = tag
        self.text = text
        self.depth = depth
        self.raw_depth = raw_depth  # Store unadjusted depth
        self.id = id
        self.class_names = class_names
        self.line = line
        self.xpath = xpath
        self._html = html.strip() if html else ""

    def get_html(self) -> str:
        """
        Retrieves the HTML content of the node.

        Returns:
            The HTML content as a string if set, otherwise empty string.
        """
        return self._html

    def get_node(self, node_id: str) -> Optional['BaseNode']:
        """
        Retrieves a node by its ID. BaseNode implementation returns self if ID matches.

        Args:
            node_id: The ID of the node to find.

        Returns:
            The BaseNode with the matching ID, or None if not found.
        """
        return self if self.id == node_id else None


class TreeNode(BaseNode):
    """A node representing an HTML element with its children."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        class_names: List[str] = [],
        children: Optional[List['TreeNode']] = None,
        line: int = 0,
        xpath: Optional[str] = None,
        html: Optional[str] = None,
    ):
        super().__init__(
            tag=tag,
            text=text,
            depth=depth,
            id=id,
            class_names=class_names,
            line=line,
            xpath=xpath,
            html=html,
            raw_depth=None
        )
        self._text = text
        self._children: List['TreeNode'] = children if children is not None else []
        self._parent_node: Optional['TreeNode'] = None
        self._links: List[str] = []
        self._initialize_links()

    def _initialize_links(self) -> None:
        """
        Initializes the _links variable by collecting unique URLs from this node's HTML and its descendants.
        """
        links = []
        html = self.get_html()
        if html:
            links.extend(scrape_links(html))
        for child in self._children:
            links.extend(child._links)
        seen = set()
        self._links = [link for link in links if not (
            link in seen or seen.add(link))]

    @property
    def parent_id(self) -> Optional[str]:
        """
        Returns the ID of the parent node if it exists, else None.

        Returns:
            The parent's ID or None.
        """
        return self._parent_node.id if self._parent_node else None

    # @property
    # def header(self) -> str:
    #     """
    #     Returns the node's header text if it's a heading tag (h1–h6), else an empty string.

    #     Returns:
    #         The header text or an empty string.
    #     """
    #     if self.tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
    #         return self._text.strip() if self._text else ""
    #     return ""

    @property
    def _content(self) -> str:
        """
        Returns the combined content of the node and its descendants, excluding header text if this is a heading node.

        Returns:
            The combined content as a string.
        """
        texts = []
        # if not self.get_header() and self._text and self._text.strip():
        #     texts.append(self._text.strip())
        for child in self._children:
            child_content = child._content
            if child_content:
                texts.append(child_content)
        return "\n".join(texts).strip()

    @property
    def text(self) -> str:
        """
        Returns the header and content joined by a newline if both are non-empty, else the original text or content.

        Returns:
            The combined header and content, or the original text, or None.
        """
        # texts = []
        # if self._text and self._text.strip():
        #     texts.append(self._text.strip())

        #     for child in self._children:
        #         child_content = child._content
        #         if child_content:
        #             texts.append(child_content)
        # return "\n".join(texts).strip()
        return self._text or ""

    @text.setter
    def text(self, value: Optional[str]) -> None:
        """
        Sets the internal text value.

        Args:
            value: The text value to set.
        """
        self._text = value

    @property
    def level(self) -> Optional[int]:
        """
        Returns the heading level (1–6) if the node is a heading tag, else None.

        Returns:
            The heading level or None.
        """
        if self.tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return int(self.tag.lower()[1])
        return None

    # @property
    # def parent_header(self) -> Optional[str]:
    #     """
    #     Returns the header text of the parent node if it exists, else None.

    #     Returns:
    #         The parent's header text or None.
    #     """
    #     parent = self.get_parent_node()
    #     return parent.header if parent else None

    # @property
    # def parent_content(self) -> Optional[str]:
    #     """
    #     Returns the content of the parent node if it exists, else None.

    #     Returns:
    #         The parent's content or None.
    #     """
    #     parent = self.get_parent_node()
    #     return parent.content if parent else None

    @property
    def parent_level(self) -> Optional[int]:
        """
        Returns the heading level of the parent node if it exists, else None.

        Returns:
            The parent's heading level or None.
        """
        parent = self.get_parent_node()
        return parent.level if parent else None

    # @property
    # def parent_headers(self) -> List[str]:
    #     """
    #     Returns a list of header texts from parent nodes with lower levels, ordered from root to immediate parent.

    #     Returns:
    #         A list of parent header texts.
    #     """
    #     headers = []
    #     current = self.get_parent_node()
    #     while current:
    #         if current.header and current.level is not None and (self.level is None or current.level < self.level):
    #             headers.append((current.header, current.level))
    #         current = current.get_parent_node()
    #     headers.sort(key=lambda x: x[1])
    #     return [header for header, _ in headers]

    def get_parent_node(self) -> Optional['TreeNode']:
        """
        Retrieves the parent node stored in the _parent_node attribute.

        Returns:
            The parent TreeNode if set, otherwise None.
        """
        return self._parent_node

    def get_node(self, node_id: str) -> Optional[Union['TreeNode', 'BaseNode']]:
        """
        Retrieves a node by its ID from the tree, searching recursively through children.

        Args:
            node_id: The ID of the node to find.

        Returns:
            The TreeNode or BaseNode with the matching ID, or None if not found.
        """
        if self.id == node_id:
            return self
        for child in self._children:
            result = child.get_node(node_id)
            if result is not None:
                return result
        return None

    def get_content(self) -> str:
        content = self.text or ""
        for child in self._children:
            content += "\n" + child.get_content()
            content = content.strip()
        return content

    def get_header(self) -> str:
        """
        Returns the node's header text if it's a heading tag (h1–h6), else an empty string.

        Returns:
            The header text or an empty string.
        """
        if self.tag.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return self._text.strip() if self._text else ""
        return ""

    def has_children(self) -> bool:
        """
        Checks if the node has any children.

        :return: True if the node has children, False otherwise.
        """
        return len(self._children) > 0

    def get_children(self) -> List['TreeNode']:
        """
        Returns the list of child nodes.

        :return: A list of TreeNode objects representing the children.
        """
        return self._children

    def get_links(self) -> List[str]:
        """
        Returns a list of all unique URLs found in this node's outer HTML and its descendants' outer HTML.

        :return: A list of unique URL strings.
        """
        return self._links

    @property
    def children(self) -> List['TreeNode']:
        """
        Public read-only access to the node's children.
        """
        return self._children.copy()


def create_node(node: TreeNode) -> TreeNode:
    """
    Creates a TreeNode from a TreeNode, copying all relevant attributes.

    Args:
        node: The source TreeNode to copy attributes from.

    Returns:
        A new TreeNode instance with attributes copied from the source TreeNode.
    """
    tree_node = TreeNode(
        tag=node.tag,
        text=node.text,
        depth=node.depth,
        id=node.id,
        class_names=node.class_names,
        children=node._children,
        line=node.line,
        xpath=node.xpath,
        html=node.get_html()
    )
    tree_node._parent_node = node._parent_node
    return tree_node


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
    timeout_ms: int = 10000,
    with_screenshot: bool = True,
    wait_for_js: bool = True,
    headless: bool = True,
) -> TreeNode:
    """
    Extracts a tree structure from HTML with id, parent_id, links attributes, actual line numbers, and outer HTML from formatted HTML.
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
        traces_dir = f"{get_entry_file_dir()}/generated/{get_entry_file_name(remove_extension=True)}/playwright/traces"
        os.makedirs(traces_dir, exist_ok=True)
        browser = p.chromium.launch(
            headless=headless,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            traces_dir=traces_dir,
        )

        ua = UserAgent()
        page = browser.new_page(user_agent=ua.random)

        if url:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        else:
            page.set_content(html, timeout=timeout_ms, wait_until="domcontentloaded")

        if wait_for_js:
            js_timeout = 5000
            logger.debug(f"Waiting JS content for {js_timeout // 1000}s")
            page.wait_for_timeout(js_timeout)

        if with_screenshot:
            # Generate a random screenshot name using uuid
            screenshot_name = f"screenshot_{uuid.uuid4().hex}.png"
            screenshot_path = f"{get_entry_file_dir()}/generated/{get_entry_file_name(remove_extension=True)}/playwright/screenshots/{screenshot_name}"
            screenshot = page.screenshot(full_page=True, path=screenshot_path)
            if screenshot:
                decoded_screenshot = base64.b64encode(screenshot).decode('utf-8')
                logger.debug(f"Decoded screenshot, length: {len(decoded_screenshot)}")
                logger.success(f"Screenshot saved at: {screenshot_path}")

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
        class_names=[],
        children=[],
        line=root_line,
        html=pq(root_el).outer_html()
    )
    if root_node._parent_node is None:
        root_node._parent_node = None

    # Add XPath for root node
    root_node.xpath = get_xpath(root_el)

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

            # Add XPath for child node
            xpath = get_xpath(child)
            child_node = TreeNode(
                tag=tag,
                text=text,
                depth=depth + 1,
                id=element_id,
                class_names=class_names,
                children=[],
                line=line_number,
                xpath=xpath,
                html=child_pq.outer_html()
            )
            if child_node._parent_node is None:
                child_node._parent_node = parent_node

            # Check if child text is a substring of parent text and empty parent text if true
            if parent_node.text and child_node.text and child_node.text in parent_node.text:
                parent_node.text = ""

            parent_node._children.append(child_node)

            # Only traverse deeper if the tag is not in TEXT_ELEMENTS
            if tag.lower() not in TEXT_ELEMENTS and child_pq.children():
                stack.append((child, child_node, depth + 1))

    return root_node


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

    Args:
        source: The HTML source string or URL.
        tags_to_split_on: List of tuples mapping markdown prefixes to HTML heading tags.
        excludes: List of tags to exclude from the tree.

    Returns:
        A list of TreeNode objects representing heading hierarchies.
    """
    results: List[TreeNode] = []
    parent_stack: List[Tuple[int, TreeNode]] = []
    seen_ids: Set[str] = set()

    def clone_node(node: TreeNode, parent_node: Optional[TreeNode] = None, new_depth: int = 0) -> TreeNode:
        """
        Clones a node with a new parent and depth, preserving structure without duplicating IDs.
        Sets _parent_node if not already set.

        Args:
            node: The TreeNode to clone.
            parent_node: The parent TreeNode, if any.
            new_depth: The depth for the cloned node.

        Returns:
            A new TreeNode with copied attributes.
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
            class_names=node.class_names,
            children=[],
            line=node.line,
            xpath=node.xpath,
            html=node.get_html()
        )
        if cloned._parent_node is None:
            cloned._parent_node = parent_node
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

            heading_node = clone_node(node, parent_node, depth)
            results.append(heading_node)
            seen_ids.add(heading_node.id)
            parent_stack.append((level, heading_node))
        else:
            if parent_stack:
                parent_node = parent_stack[-1][1]
                child_node = clone_node(
                    node, parent_node, parent_node.depth + 1)
                parent_node._children.append(child_node)
                seen_ids.add(child_node.id)

        for child in node._children:
            traverse(child)

    tree = extract_tree_with_text(source, excludes=excludes)
    if tree:
        traverse(tree)

    return results


def extract_text_elements(
    source: str, 
    excludes: list[str] = ["nav", "footer", "script", "style"], 
    timeout_ms: int = 10000,
    with_screenshot: bool = True,
    wait_for_js: bool = True,
    headless: bool = True,
) -> List[str]:
    """
    Extracts a flattened list of text elements from the HTML document, ignoring specific elements.
    Uses Playwright to render dynamic content with advanced browser automation.

    :param source: The HTML string, file path, or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
    :param timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
    :param with_screenshot: Whether to capture screenshots during rendering.
    :param wait_for_js: Whether to wait for JavaScript execution to complete.
    :param headless: Whether to run browser in headless mode.
    :return: A flattened list of text elements found in the HTML.
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
        traces_dir = f"{get_entry_file_dir()}/playwright/traces"
        os.makedirs(traces_dir, exist_ok=True)
        
        browser = p.chromium.launch(
            headless=headless,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            traces_dir=traces_dir,
        )
        
        ua = UserAgent()
        page = browser.new_page(user_agent=ua.random)
        
        if url:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        else:
            page.set_content(html, timeout=timeout_ms, wait_until="domcontentloaded")
        
        if wait_for_js:
            js_timeout = 5000
            logger.debug(f"Waiting JS content for {js_timeout // 1000}s")
            page.wait_for_timeout(js_timeout)
        
        if with_screenshot:
            # Generate a random screenshot name using uuid
            screenshot_name = f"screenshot_{uuid.uuid4().hex}.png"
            screenshot_path = f"{get_entry_file_dir()}/generated/{get_entry_file_name(remove_extension=True)}/playwright/screenshots/{screenshot_name}"
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            screenshot = page.screenshot(full_page=True, path=screenshot_path)
            if screenshot:
                decoded_screenshot = base64.b64encode(screenshot).decode('utf-8')
                logger.debug(f"Decoded screenshot, length: {len(decoded_screenshot)}")
                logger.success(f"Screenshot saved at: {screenshot_path}")
        
        page_content = page.content()
        browser.close()
    
    # Format the HTML content using format_html
    formatted_html = format_html(page_content)
    doc = pq(formatted_html)
    exclude_elements(doc, excludes)
    
    def extract_text(element, depth: int = 0) -> List[str]:
        """Recursively extract text from elements, handling special cases."""
        el_pq = pq(element)
        
        # Skip comment nodes
        if element.tag is Comment or str(element.tag).startswith('<cyfunction Comment'):
            return []
        
        tag = element.tag if isinstance(element.tag, str) else str(element.tag)
        
        # Handle special text extraction for certain tags
        if tag.lower() == "meta":
            text = el_pq.attr("content") or ""
            text = decode_text_with_unidecode(text.strip())
            if text:
                return [text]
            return []
        elif tag.lower() in TEXT_ELEMENTS:
            # For text elements, extract direct text content
            text_parts = []
            for item in el_pq.contents():
                if isinstance(item, str):
                    text_parts.append(item.strip())
                else:
                    text_parts.append(decode_text_with_unidecode(
                        pq(item).text().strip()))
            text = " ".join(part for part in text_parts if part)
        else:
            text = decode_text_with_unidecode(el_pq.text().strip())
        
        text_elements = []
        
        # If element has meaningful text and no children, return it directly
        if text and len(el_pq.children()) == 0:
            text_elements.append(text)
        else:
            # Recursively extract from children for non-text elements
            if tag.lower() not in TEXT_ELEMENTS:
                for child in el_pq.children():
                    text_elements.extend(extract_text(child, depth + 1))
        
        return text_elements
    
    # Start extraction from root element
    text_elements = extract_text(doc[0])
    return text_elements


def flatten_tree_to_base_nodes(root: TreeNode) -> List[TreeNode]:
    """
    Flattens a TreeNode hierarchy into a list of TreeNode objects.

    Args:
        root: The root TreeNode to flatten.

    Returns:
        A list of TreeNode objects representing all nodes in the tree in depth-first order.
    """
    result: List[TreeNode] = []

    def traverse(node: TreeNode) -> None:
        node_copy = create_node(node)
        # Remove 'children' property from the copy before appending
        if hasattr(node_copy, '_children'):
            node_copy._children = []
        result.append(node_copy)

        # Recursively process children
        for child in node.get_children():
            traverse(child)

    traverse(root)
    return result


def get_leaf_nodes(root: TreeNode) -> List[TreeNode]:
    """
    Returns a list of TreeNode objects for leaf nodes (nodes with no children).

    Args:
        root: The root TreeNode of the tree to traverse.

    Returns:
        A list of TreeNode objects for leaf nodes.
    """
    result: List[TreeNode] = []

    def traverse(node: TreeNode) -> None:
        # Check if the node is a leaf (no children)
        if not node.has_children():
            result.append(create_node(node))

        # Recursively traverse children
        for child in node.get_children():
            traverse(child)

    traverse(root)
    return result


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
            has_child_text = node._children and node._children[0].text

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

            for child in node._children:
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
        "yahoo",
    ]

    try:
        results: list[SearchResult] = search_searxng(
            query_url="http://jethros-macbook-air.local:3000/search",
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
    except NoResultsFoundError:
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


class SignificantNode(TreeNode):
    """A node representing a significant element with its outer HTML, extending TreeNode."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        parent_id: Optional[str] = None,
        class_names: List[str] = [],
        link: Optional[str] = None,
        children: List['TreeNode'] = [],
        line: int = 0,
        html: Optional[str] = None,
        has_significant_descendants: bool = False
    ):
        super().__init__(
            tag=tag,
            text=text,
            depth=depth,
            id=id,
            class_names=class_names,
            children=children,
            line=line,
            html=html
        )
        self.has_significant_descendants = has_significant_descendants


def _node_to_outer_html(node: TreeNode) -> str:
    """
    Converts a TreeNode to its outer HTML representation, including all nested children.

    Args:
        node: The TreeNode to convert.

    Returns:
        The outer HTML string for the node and its descendants.
    """
    if node.get_html():
        return node.get_html()

    attributes = []
    if node.id:
        attributes.append(f'id="{node.id}"')
    if node.class_names:
        attributes.append(f'class="{" ".join(node.class_names)}"')

    attr_str = " ".join(attributes).strip()
    tag_open = f"<{node.tag.lower()}" + (f" {attr_str}" if attr_str else "")

    void_elements = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"
    }

    if node.tag.lower() in void_elements:
        return f"{tag_open} />"

    content = node.text or ""
    for child in node._children:
        content += _node_to_outer_html(child)

    return f"{tag_open}>{content}</{node.tag.lower()}>"


def create_significant_node(
    node: TreeNode,
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
    return SignificantNode(
        tag=node.tag,
        text=node.text,
        depth=node.depth,
        id=node.id,
        class_names=node.class_names,
        children=node._children,
        line=node.line,
        html=html,
        has_significant_descendants=has_significant_descendants
    )


def get_significant_nodes(root: TreeNode) -> List[SignificantNode]:
    """
    Returns a list of SignificantNode objects for nodes that either have a non-auto-generated ID
    or are one of the specified tags: footer, aside, header, main, nav, article, section.
    Each node includes its full outer HTML including all nested children and has_significant_descendants flag
    indicating if it has significant descendants. The parent_id field references the nearest significant ancestor,
    explicitly set to None for the root significant node. Nodes without an ID attribute are assigned a generated
    unique ID based on their tag and a counter to handle multiple instances.

    Args:
        root: The root TreeNode of the tree to traverse.

    Returns:
        A list of SignificantNode objects for matching nodes.
    """
    significant_tags = {"footer", "aside", "header",
                        "main", "nav", "article", "section"}
    result: List[SignificantNode] = []
    id_counter = {}  # Track counts for generated IDs per tag

    def traverse(node: TreeNode, significant_ancestor_id: Optional[str]) -> bool:
        is_significant = (node.id and not node.id.startswith(
            "auto_")) or node.tag.lower() in significant_tags

        node_id = node.id
        if is_significant and (node.id is None or node.id == ""):
            tag = node.tag.lower()
            id_counter[tag] = id_counter.get(tag, 0) + 1
            node_id = f"generated_{tag}_{id_counter[tag]}"

        current_significant_ancestor_id = node_id if is_significant and node_id and not node_id.startswith(
            "auto_") else significant_ancestor_id

        has_significant_descendants = False
        for child in node._children:
            if traverse(child, current_significant_ancestor_id):
                has_significant_descendants = True

        if is_significant:
            html = _node_to_outer_html(node)
            result.append(create_significant_node(
                node=node,
                html=html,
                has_significant_descendants=has_significant_descendants
            ))

        return is_significant or has_significant_descendants

    traverse(root, None)
    return result


class ParentWithCommonClass(TreeNode):
    """Represents a parent node with its shared class name and/or tag and matching children, extending TreeNode."""

    def __init__(
        self,
        tag: str,
        text: Optional[str],
        depth: int,
        id: str,
        class_names: List[str] = [],
        link: Optional[str] = None,
        children: List['TreeNode'] = [],
        line: int = 0,
        html: Optional[str] = None,
        common_class: str = "",
        common_tag: str = "",
        children_count: int = 0
    ):
        super().__init__(
            tag=tag,
            text=text,
            depth=depth,
            id=id,
            class_names=class_names,
            children=children,
            line=line,
            html=html
        )
        self.common_class = common_class
        self.common_tag = common_tag
        self.children_count = children_count


def get_parents_with_common_class(
    root: TreeNode,
    class_name: Optional[str] = None,
    match_type: Literal["tag", "class", "both"] = "both"
) -> List[ParentWithCommonClass]:
    """
    Finds parent nodes with direct children sharing a specified or common class, tag, or both, excluding parents with children having different classes/tags.

    Args:
        root: The root TreeNode of the tree to search.
        class_name: Optional class name to match in child nodes. If provided, matches only by class, ignoring match_type.
        match_type: Determines if children must share tags ("tag"), classes ("class"), or both ("both"). Defaults to "both".

    Returns:
        A list of ParentWithCommonClass objects sorted by number of matching children (descending).
    """
    result = []

    def traverse(node: TreeNode):
        if not node.has_children():
            return

        children = node.get_children()

        if class_name:
            matching_children = [
                child for child in children if class_name in child.class_names]
            has_different_class = any(
                child.class_names and class_name not in child.class_names for child in children)
            if matching_children and not has_different_class:
                result.append(ParentWithCommonClass(
                    tag=node.tag,
                    text=node.text,
                    depth=node.depth,
                    id=node.id,
                    class_names=node.class_names,
                    line=node.line,
                    html=node.get_html(),
                    common_class=class_name,
                    common_tag="",
                    children_count=len(matching_children),
                    children=matching_children
                ))
        else:
            child_classes = {}
            child_tags = {}
            for child in children:
                for cls in child.class_names:
                    child_classes[cls] = child_classes.get(cls, 0) + 1
                child_tags[child.tag] = child_tags.get(child.tag, 0) + 1

            if match_type == "both":
                for cls, cls_count in child_classes.items():
                    if cls_count <= 1:
                        continue
                    for tag, tag_count in child_tags.items():
                        if tag_count <= 1:
                            continue
                        has_different = any(
                            (child.class_names and cls not in child.class_names) or
                            child.tag != tag
                            for child in children
                        )
                        if not has_different:
                            matching_children = [
                                child for child in children
                                if cls in child.class_names and child.tag == tag
                            ]
                            if matching_children:
                                result.append(ParentWithCommonClass(
                                    tag=node.tag,
                                    text=node.text,
                                    depth=node.depth,
                                    id=node.id,
                                    class_names=node.class_names,
                                    line=node.line,
                                    html=node.get_html(),
                                    common_class=cls,
                                    common_tag=tag,
                                    children_count=len(matching_children),
                                    children=matching_children
                                ))
            elif match_type == "class":
                for cls, count in child_classes.items():
                    if count > 1:
                        has_different_class = any(
                            child.class_names and cls not in child.class_names
                            for child in children
                        )
                        if not has_different_class:
                            matching_children = [
                                child for child in children if cls in child.class_names]
                            result.append(ParentWithCommonClass(
                                tag=node.tag,
                                text=node.text,
                                depth=node.depth,
                                id=node.id,
                                class_names=node.class_names,
                                line=node.line,
                                html=node.get_html(),
                                common_class=cls,
                                common_tag="",
                                children_count=len(matching_children),
                                children=matching_children
                            ))
            elif match_type == "tag":
                for tag, count in child_tags.items():
                    if count > 1:
                        has_different_tag = any(
                            child.tag != tag for child in children
                        )
                        if not has_different_tag:
                            matching_children = [
                                child for child in children if child.tag == tag]
                            result.append(ParentWithCommonClass(
                                tag=node.tag,
                                text=node.text,
                                depth=node.depth,
                                id=node.id,
                                class_names=node.class_names,
                                line=node.line,
                                html=node.get_html(),
                                common_class="",
                                common_tag=tag,
                                children_count=len(matching_children),
                                children=matching_children
                            ))

        for child in children:
            traverse(child)

    traverse(root)
    result.sort(key=lambda x: x.children_count, reverse=True)
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
    "get_parents_with_common_class",
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

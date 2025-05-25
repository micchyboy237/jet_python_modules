from datetime import datetime
from jet.wordnet.sentence import split_sentences
from lxml.etree import Comment
from typing import Callable, Optional, List, Dict, TypedDict, Union
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


def scrape_links(html: str, base_url: Optional[str] = None) -> List[str]:
    # Target attributes to extract
    attributes = ['href', 'data-href', 'action']

    # Build the pattern dynamically to support quoted values (single or double)
    attr_pattern = '|'.join(attributes)
    quote_pattern = (
        rf'(?:{attr_pattern})\s*=\s*'      # attribute and equal sign
        r'(["\'])'                         # opening quote (capture group 1)
        r'(.*?)'                           # value (capture group 2)
        r'\1'                              # matching closing quote
    )

    matches = re.findall(quote_pattern, html, flags=re.IGNORECASE)

    # Filter and process links
    filtered = []
    for match in matches:
        link = match[1].strip()
        # Skip empty, javascript:, or invalid links
        if not link or link.lower().startswith('javascript:'):
            continue

        if base_url:
            # Parse base_url to get scheme and netloc for filtering
            parsed_base = urlparse(base_url)
            base_scheme_netloc = f"{parsed_base.scheme}://{parsed_base.netloc}"

            # Resolve relative URLs and anchor links
            if link.startswith('#'):
                # Prepend base_url's scheme, netloc, and path to anchor links
                link = f"{base_scheme_netloc}{parsed_base.path}{link}"
            else:
                # Resolve relative URLs against base_url
                link = urljoin(base_url, link)

            # Filter links from the same domain (scheme and netloc)
            if urlparse(link).netloc != parsed_base.netloc:
                continue
        else:
            # If no base_url, filter out fragment-only links and links without a host
            if link == '#' or link.startswith('#'):
                continue
            parsed_link = urlparse(link)
            if not (parsed_link.scheme and parsed_link.netloc):
                continue

        filtered.append(link)

    # Return unique links only
    return list(set(filtered))


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


def clean_punctuations(content: str) -> str:
    """
    Replace consecutive and mixed punctuation marks (.?!), ensuring that each valid group 
    is replaced with its last occurring punctuation.

    Example:
        "Hello!!! How are you???" -> "Hello! How are you?"
        "Wait... What.!?" -> "Wait. What?"
        "Really...?!? Are you sure???" -> "Really. Are you sure?"

    :param content: Input string with possible consecutive punctuations.
    :return: String with cleaned punctuation.
    """
    return re.sub(r'([.?!#-)]+)', lambda match: match.group()[-1], content)


def protect_links(text: str) -> (str, List[str]):
    # Find all markdown links and replace them with placeholders
    links = re.findall(r'\[.*?\]\(.*?\)', text)
    for i, link in enumerate(links):
        text = text.replace(link, f"__LINK_{i}__")
    return text, links


def restore_links(text: str, links: List[str]) -> str:
    for i, link in enumerate(links):
        text = text.replace(f"__LINK_{i}__", link)
    return text


def clean_spaces(content: str) -> str:
    content, links = protect_links(content)

    # Remove spaces before .?!,;:])}
    content = re.sub(r'\s*([.?!,;:\]\)}])', r'\1', content)

    # Ensure single space *after* punctuation if followed by alphanum
    content = re.sub(r'([.?!,;:\]\)}])(\w)', r'\1 \2', content)

    # Remove consecutive spaces
    content = re.sub(r' +', ' ', content).strip()

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


class TreeNode:
    def __init__(self, tag: str, text: Optional[str], depth: int, id: str,
                 parent: Optional[str], class_names: List[str] = [],
                 link: Optional[str] = None, children: Optional[List['TreeNode']] = None):
        self.tag = tag
        self.text = text
        self.depth = depth
        self.id = id
        self.parent = parent
        self.class_names = class_names
        self.link = link
        self.children: List['TreeNode'] = children if children is not None else [
        ]

    def get_content(self) -> str:
        content = self.text or ""
        for child in self.children:
            content += child.get_content()
        return content.strip()


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
    excludes: list[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 1000
) -> Optional[TreeNode]:
    """
    Extracts a tree structure from HTML with id, parent, and link attributes on each node.
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

    doc = pq(page_content)
    exclude_elements(doc, excludes)

    root_el = doc[0]
    root_id = f"auto_{uuid.uuid4().hex[:8]}"
    tag_name = root_el.tag if isinstance(
        root_el.tag, str) else str(root_el.tag)

    root_node = TreeNode(
        tag=tag_name,
        text=None,
        depth=0,
        id=root_id,
        parent=None,
        class_names=[],
        link=None,
        children=[]
    )

    stack = [(root_el, root_node, 0)]

    while stack:
        el, parent_node, depth = stack.pop()
        el_pq = pq(el)

        for child in el_pq.children():
            child_pq = pq(child)
            if child.tag is Comment or str(child.tag).startswith('<cyfunction Comment'):
                continue

            tag = child.tag if isinstance(child.tag, str) else str(child.tag)
            text = decode_text_with_unidecode(child.text)
            class_names = [cls for cls in (child_pq.attr(
                "class") or "").split() if not cls.startswith("css-")]
            element_id = child_piselement_id = child_pq.attr("id")

            link = (child_pq.attr("href") or
                    child_pq.attr("data-href") or
                    child_pq.attr("action") or
                    None)

            if not element_id or not re.match(r'^[a-zA-Z_-]+$', element_id):
                element_id = f"auto_{uuid.uuid4().hex[:8]}"

            child_node = TreeNode(
                tag=tag,
                text=text or "",
                depth=depth + 1,
                id=element_id,
                parent=parent_node.id,
                class_names=class_names,
                link=link,
                children=[]
            )

            parent_node.children.append(child_node)

            if child_pq.children():
                stack.append((child, child_node, depth + 1))

    return root_node


def extract_by_heading_hierarchy(
    source: str,
    tags_to_split_on: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ],
    excludes: list[str] = ["nav", "footer", "script", "style"]
) -> List[TreeNode]:
    """
    Extracts a list of TreeNode hierarchies split by heading tags, avoiding duplicates,
    with heading text prepended by '#' based on header level.
    """
    results: List[TreeNode] = []
    parent_stack: List[Tuple[int, TreeNode]] = []
    seen_ids: set = set()

    def clone_node(node: TreeNode, new_parent_id: Optional[str] = None, new_depth: int = 0) -> TreeNode:
        """
        Clones a node with a new parent and depth, preserving structure without duplicating IDs.
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
            parent=new_parent_id,
            class_names=node.class_names,
            link=node.link,
            children=[]
        )
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


class TextHierarchyResult(TypedDict):
    text: str
    links: List[str]
    depth: int
    id: str
    parent: Optional[str]
    parent_text: Optional[str]


def extract_texts_by_hierarchy(
    source: str,
    tags_to_split_on: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ],
    ignore_links: bool = True,
    excludes: list[str] = [
        "nav", "footer", "script", "style", "noscript", "template",
        "svg", "canvas", "iframe", "form", "input", "button",
        "select", "option", "label", "aside", "meta", "link",
        "figure", "figcaption", "object", "embed"
    ]
) -> List[TextHierarchyResult]:
    """
    Extracts a list of dictionaries from HTML, each containing the combined text of a heading
    and its descendants, a list of unique links, depth, id, parent, and parent_text attributes.

    Args:
        source: HTML string, URL, or file path to process.
        tags_to_split_on: List of tuples with (prefix, tag) to split the hierarchy (e.g., [("#", "h1"), ("##", "h2")]).
        ignore_links: If True, excludes text from nodes with 'a' tags but includes their links.
        excludes: List of HTML tags whose text and links should be excluded (e.g., ["footer", "nav"]).

    Returns:
        List of dictionaries, each with 'text' (combined text of heading and descendants),
        'links' (list of unique links), 'depth' (node depth), 'id' (node ID), 
        'parent' (parent node ID or None), and 'parent_text' (combined text of parent and its descendants or None).
    """
    def collect_text_and_links(node: TreeNode) -> Tuple[TextHierarchyResult, str]:
        """
        Recursively collects text, unique links, depth, id, parent, and parent_text from a node and its children,
        and returns the combined text for the node.
        """
        texts = []
        links = set()

        if node.text and node.text.strip():
            if not (ignore_links and node.link):
                texts.append(node.text.strip())
        if node.link:
            links.add(node.link)

        for child in node.children:
            child_result, child_text = collect_text_and_links(child)
            if child_text:
                texts.append(child_text)
            links.update(child_result["links"])

        combined_text = "\n".join(text for text in texts if text)

        return {
            "text": combined_text,
            "links": list(links),
            "depth": node.depth,
            "id": node.id,
            "parent": node.parent,
            "parent_text": None  # Will be populated later using id_to_text
        }, combined_text

    heading_nodes = extract_by_heading_hierarchy(
        source, tags_to_split_on, excludes)

    # Collect full text (including descendants) for each heading node
    id_to_text = {}
    results = []
    for node in heading_nodes:
        result, combined_text = collect_text_and_links(node)
        id_to_text[node.id] = combined_text
        results.append(result)

    # Populate parent_text using id_to_text
    for result in results:
        if result["parent"]:
            result["parent_text"] = id_to_text.get(result["parent"], None)

    return results


class MergedTextsResult(TypedDict):
    text: str
    token_count: int


def merge_texts_by_hierarchy(
    source: str,
    tokenizer: Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]],
    max_tokens: int,
    tags_to_split_on: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ],
    ignore_links: bool = True,
    excludes: list[str] = [
        "nav", "footer", "script", "style", "noscript", "template",
        "svg", "canvas", "iframe", "form", "input", "button",
        "select", "option", "label", "aside", "meta", "link",
        "figure", "figcaption", "object", "embed"
    ],
    split_fn: Optional[Callable[[str], List[str]]] = None
) -> List[MergedTextsResult]:
    from jet.code.splitter_markdown_utils import get_md_header_contents

    # Extract texts with hierarchy
    results = get_md_header_contents(
        source, headers_to_split_on=tags_to_split_on, ignore_links=ignore_links)
    texts = [result["content"] for result in results]

    # Initialize variables for grouping
    grouped_texts: List[str] = []
    current_group: List[str] = []
    current_token_count: int = 0

    for i, (result, text) in enumerate(zip(results, texts)):
        # Skip empty texts
        if not text.strip():
            continue

        # Tokenize the current text
        tokenized_text = tokenizer(text) if isinstance(
            tokenizer(text), list) else tokenizer(text)[0]
        token_count = len(tokenized_text)

        # Get parent header for context
        parent_header = result["header"]

        # If single text exceeds max_tokens, split and handle parts
        if token_count > max_tokens:
            sentences: List[str] = split_fn(
                text) if split_fn else split_sentences(text)
            for split_n, sentence in enumerate(sentences):
                sentence_tokens = tokenizer(sentence) if isinstance(
                    tokenizer(sentence), list) else tokenizer(sentence)[0]
                sentence_token_count = len(sentence_tokens)

                if current_token_count + sentence_token_count <= max_tokens:
                    current_group.append(sentence)
                    current_token_count += sentence_token_count
                else:
                    if current_group:
                        grouped_texts.append("\n".join(current_group))
                        current_group = []
                        current_token_count = 0
                    if sentence_token_count <= max_tokens:
                        # Prepend header with part number only when splitting
                        header_with_part = f"{parent_header} - {split_n + 1}" if parent_header else None
                        current_group.append(
                            (header_with_part + " " + sentence) if header_with_part else sentence)
                        current_token_count = sentence_token_count
        else:
            # If text fits within max_tokens, try to add to current group
            if current_token_count + token_count <= max_tokens:
                current_group.append(text)
                current_token_count += token_count
            else:
                if current_group:
                    grouped_texts.append("\n".join(current_group))
                    current_group = [text]
                    current_token_count = token_count
                else:
                    current_group = [text]
                    current_token_count = token_count

        # Try merging with next text if possible
        if i + 1 < len(texts) and current_token_count > 0:
            next_text = texts[i + 1]
            if not next_text.strip():
                continue
            next_tokenized = tokenizer(next_text) if isinstance(
                tokenizer(next_text), list) else tokenizer(next_text)[0]
            next_token_count = len(next_tokenized)

            if current_token_count + next_token_count <= max_tokens:
                merged_text = " ".join(current_group) + " " + next_text
                merged_tokens = tokenizer(merged_text) if isinstance(
                    tokenizer(merged_text), list) else tokenizer(merged_text)[0]
                if len(merged_tokens) <= max_tokens:
                    current_group.append(next_text)
                    current_token_count += next_token_count
                    texts[i + 1] = ""  # Mark as processed

    # Add final group if it exists
    if current_group:
        grouped_texts.append("\n".join(current_group))

    # Filter out empty strings
    filtered_texts = [text.strip() for text in grouped_texts if text.strip()]
    batched_token_texts: List[List[str]] = tokenizer(filtered_texts)
    token_counts: List[int] = [len(token_texts)
                               for token_texts in batched_token_texts]

    merged_texts = []
    for text, token_count in zip(filtered_texts, token_counts):
        merged_texts.append({"text": text, "token_count": token_count})

    return merged_texts


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


def search_data(query, **kwargs) -> list[SearchResult]:
    filter_sites = []
    engines = [
        "google",
        "brave",
        "duckduckgo",
        "bing",
        "yahoo",
    ]

    # Simulating the search function with the placeholder for your search logic
    results: list[SearchResult] = search_searxng(
        query_url="http://jetairm1:3000/search",
        query=query,
        filter_sites=filter_sites,
        engines=engines,
        config={
            "port": 3101
        },
        **kwargs
    )

    if not results:
        raise NoResultsFoundError(f"No results found for query: '{query}'")

    return results


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

from copy import deepcopy
import uuid
from jet.search.formatters import decode_text_with_unidecode
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from collections import defaultdict
import os
from typing import Generator, List, Optional
from urllib.parse import urljoin, urlparse
from jet.search.searxng import NoResultsFoundError, search_searxng, SearchResult
from pyquery import PyQuery as pq
from jet.logger.config import colorize_log
from jet.logger import logger
from typing import List, Dict, Optional
import json
import re
import string
from jet.utils.text import fix_and_unidecode
import parsel
import unidecode


def scrape_links(html: str) -> List[str]:
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

    # Filter out empty, '#' and javascript: links
    filtered = [
        match[1] for match in matches
        if match[1].strip() not in ('', '#') and not match[1].lower().startswith('javascript:')
    ]
    return filtered


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
    tags_to_exclude = ["style", "script", "nav", "header", "footer",
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


def clean_newlines(content, max_newlines: int = 3) -> str:
    """Merge consecutive newlines from the content, but limit to at most max_newlines consecutive newlines."""
    # Remove trailing whitespace for each line
    content = '\n'.join([line.rstrip() for line in content.split('\n')])

    if max_newlines == 0:
        # Replace all consecutive newlines with a single space
        content = re.sub(r'\n+', ' ', content)
    else:
        # Reduce consecutive newlines to at most max_newlines newlines
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


def clean_spaces(content: str) -> str:
    # Remove spaces before .?!,;:\]\)}
    content = re.sub(r'\s*([.?!,;:\]\)}])', r'\1', content)

    # Ensure single spacing on the right of .?!,;:\]\)} only if the next character is alphanumeric
    content = re.sub(r'([.?!,;:\]\)}])(\w)', r'\1 \2', content)

    # Remove consecutive spaces
    content = re.sub(r' +', ' ', content).strip()

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


class TitleMetadata(TypedDict):
    title: Optional[str]
    metadata: Dict[str, str]


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


def exclude_elements(doc, excludes: List[str]) -> None:
    """
    Removes elements from the document that match the tags in the excludes list.

    :param doc: The PyQuery object representing the HTML document.
    :param excludes: A list of tag names to exclude (e.g., ["style", "script"]).
    """
    for tag in excludes:
        # Remove all elements of the excluded tag from the document
        for element in doc(tag):
            pq(element).remove()


class TreeNode:
    def __init__(self, tag: str, text: Optional[str], depth: int, id: str,
                 parent: Optional[str], class_names: List[str] = [], children: Optional[List['TreeNode']] = None):
        self.tag = tag
        self.text = text
        self.depth = depth
        self.id = id
        self.parent = parent
        self.class_names = class_names
        self.children: List['TreeNode'] = children if children is not None else [
        ]

    def get_content(self) -> str:
        content = self.text or ""
        for child in self.children:
            content += child.get_content()
        return content.strip()


def add_child_nodes(flat_nodes: List[TreeNode]) -> List[TreeNode]:
    """
    Given a flat list of TreeNodes with parent references,
    this reconstructs the hierarchy by assigning children accordingly.
    """
    id_map: Dict[str, TreeNode] = {node.id: node for node in flat_nodes}
    root_nodes: List[TreeNode] = []

    for node in flat_nodes:
        if node.parent and node.parent in id_map:
            parent_node = id_map[node.parent]
            parent_node.children.append(node)
        root_nodes.append(node)

    return root_nodes


def extract_tree_with_text(
    source: str,
    excludes: List[str] = ["style", "script"],
    timeout_ms: int = 1000
) -> Optional[TreeNode]:
    """
    Extracts a tree structure from HTML with id and parent attributes on each node.
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
        children=[]
    )

    stack = [(root_el, root_node, 0)]

    while stack:
        el, parent_node, depth = stack.pop()
        el_pq = pq(el)

        for child in el_pq.children():
            child_pq = pq(child)
            tag = child.tag if isinstance(child.tag, str) else str(child.tag)

            # ✅ Preserve all text, even when children exist
            # text = child_pq.text().strip()
            text = decode_text_with_unidecode(child.text)

            class_names = [cls for cls in (child_pq.attr(
                "class") or "").split() if not cls.startswith("css-")]
            element_id = child_pq.attr("id")

            if not element_id or not re.match(r'^[a-zA-Z_-]+$', element_id):
                element_id = f"auto_{uuid.uuid4().hex[:8]}"

            child_node = TreeNode(
                tag=tag,
                text=text or "",
                depth=depth + 1,
                id=element_id,
                parent=parent_node.id,
                class_names=class_names,
                children=[]
            )

            parent_node.children.append(child_node)

            if child_pq.children():
                stack.append((child, child_node, depth + 1))

    return root_node


def extract_by_heading_hierarchy(
    source: str,
    tags_to_split_on: List[str] = ["h1", "h2", "h3", "h4", "h5", "h6"]
) -> List[TreeNode]:
    results: List[TreeNode] = []
    parent_stack: List[Tuple[int, TreeNode]] = []

    def clone_subtree(node: TreeNode, new_parent_id: Optional[str] = None, new_depth: int = 0) -> TreeNode:
        cloned = TreeNode(
            tag=node.tag,
            text=node.text,
            depth=new_depth,
            id=node.id,
            parent=new_parent_id,
            class_names=node.class_names,
            children=[]
        )
        for child in node.children:
            cloned_child = clone_subtree(child, cloned.id, new_depth + 1)
            cloned.children.append(cloned_child)
        return cloned

    def traverse(node: TreeNode) -> None:
        nonlocal parent_stack

        if node.tag in tags_to_split_on:
            level = tags_to_split_on.index(node.tag)

            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_node = parent_stack[-1][1] if parent_stack else None
            depth = parent_node.depth + 1 if parent_node else 0

            heading_node = clone_subtree(
                node, parent_node.id if parent_node else None, depth)
            results.append(heading_node)
            parent_stack.append((level, heading_node))

        else:
            if parent_stack:
                parent_node = parent_stack[-1][1]
                child_node = clone_subtree(
                    node, parent_node.id, parent_node.depth + 1)
                parent_node.children.append(child_node)

        for child in node.children:
            traverse(child)

    tree = extract_tree_with_text(source)
    if tree:
        traverse(tree)

    # ✅ Rebuild child relationships
    # results_with_child_nodes = add_child_nodes(results)
    # return results_with_child_nodes
    return results


def extract_text_elements(source: str, excludes: List[str] = ["style", "script"], timeout_ms: int = 1000) -> List[str]:
    """
    Extracts a flattened list of text elements from the HTML document, ignoring specific elements like <style> and <script>.
    Uses Playwright to render dynamic content if needed.

    :param source: The HTML string or URL to parse.
    :param excludes: A list of tag names to exclude (e.g., ["style", "script"]).
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


def format_html(html: str, excludes: List[str] = ["style", "script"]) -> str:
    """
    Converts the tree-like structure of HTML elements and text into a single formatted string.
    """
    tree = extract_tree_with_text(html, excludes)

    def build_html(node: TreeNode, indent=0) -> str:
        result = ""

        if node:
            # Skip node if its tag is in the excludes list
            if node.tag in excludes:
                return result

            # Only include nodes with meaningful content or children
            has_text = bool(node.text)
            has_id = bool(node.id)
            has_class = bool(node.class_names)
            has_child_text = node.children and node.children[0].text

            if has_text or has_id or has_class or has_child_text:
                tag_text = node.tag
                if has_id:
                    tag_text += f' id="{node.id}"'
                if has_class:
                    tag_text += ' class="' + " ".join(node.class_names) + '"'

                if has_text:
                    result += '  ' * indent + \
                        f"<{tag_text}>{node.text}</{node.tag}>\n"
                else:
                    result += '  ' * indent + f"<{tag_text}>\n"

                for child in node.children:
                    result += build_html(child, indent + 1)

                if not has_text:
                    result += '  ' * indent + f"</{node.tag}>\n"
            else:
                for child in node.children:
                    result += build_html(child, indent + 1)

        return result

    return build_html(tree)


# Function to print the tree-like structure recursively
def print_html(html: str):
    tree = extract_tree_with_text(html)

    def print_tree(node: TreeNode, indent=0, excludes: List[str] = ["style", "script"]):
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

    # # Last path segment without extension
    # path_parts = [part for part in parsed.path.split('/') if part]
    # last_path = path_parts[-1] if path_parts else 'root'
    # last_path_no_ext = os.path.splitext(last_path)[0]

    # Build final safe path: output_dir / safe_host / last_path_no_ext
    # return os.path.join(output_dir, safe_host, last_path_no_ext)
    return os.path.join(output_dir, safe_host)


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


def scrape_urls(urls: list[str], *, max_depth: Optional[int] = 2, query: Optional[str] = None) -> Generator[tuple[str, str], None, None]:
    from jet.scrapers.crawler.web_crawler import WebCrawler

    crawler = WebCrawler(max_depth=max_depth, query=query)

    for url in urls:
        for result in crawler.crawl(url):
            yield result["url"], result["html"]

    crawler.close()


def validate_headers(html: str, min_count: int = 5) -> bool:
    from jet.scrapers.preprocessor import html_to_markdown
    from jet.code.splitter_markdown_utils import count_md_header_contents

    md_text = html_to_markdown(html)
    header_count = count_md_header_contents(md_text)
    return header_count >= min_count


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
    "format_html",
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

"""
HTML Skeleton Extractor for LLM Context Generation
Extracts skeletal structure from HTML documents to provide context for LLM-based
query selector generation. Implements block-tree pruning and structural compression
based on HtmlRAG methodology [[6]].
Key Features:
- Tag whitelisting to preserve semantic structure
- Structural compression (merge single-child containers)
- Empty element removal
- Text content truncation/placeholder replacement
- Configurable granularity control
- Accessibility attribute preservation
- URL compression to reduce context bloat
- Filtering of unreliable randomly-generated class names
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

DEFAULT_KEEP_TAGS = frozenset(
    [
        "html",
        "head",
        "body",
        "header",
        "footer",
        "nav",
        "main",
        "section",
        "article",
        "aside",
        "div",
        "p",
        "span",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "caption",
        "form",
        "input",
        "button",
        "select",
        "option",
        "textarea",
        "label",
        "fieldset",
        "legend",
        "a",
        "img",
        "figure",
        "figcaption",
        "picture",
        "source",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "code",
        "pre",
        "blockquote",
        "cite",
        "details",
        "summary",
        "dialog",
    ]
)
DEFAULT_REMOVE_TAGS = frozenset(
    [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "math",
        "link",
        "meta",
        "base",
        "template",
    ]
)
ESSENTIAL_ATTRIBUTES = frozenset(
    [
        "id",
        "class",
        "role",
        "aria-label",
        "aria-labelledby",
        "aria-describedby",
        "href",
        "src",
        "alt",
        "name",
        "type",
        "for",
        "data-testid",
        "data-cy",
        "data-test",
    ]
)

# Regex patterns for detecting randomly generated class names
# Matches patterns like: css-123abc, r-12345, _hash123, etc.
RANDOM_CLASS_PATTERNS = [
    re.compile(r"^css-[a-zA-Z0-9]+$"),  # Emotion/styled-components (any length)
    re.compile(
        r"^[a-z]-[a-zA-Z0-9]+$"
    ),  # Generic short-hash patterns (e.g. r-1awozwy, e8ju0x50 if prefixed)
    re.compile(r"^_[a-zA-Z0-9]+$"),  # Underscore-prefixed hashes
    re.compile(
        r"^[a-zA-Z0-9]*\d[a-zA-Z0-9]*$"
    ),  # Fallback: any class containing digits that isn't purely semantic
]

# Helper to identify "semantic" looking classes to avoid over-filtering
SEMANTIC_CLASS_PATTERN = re.compile(
    r"^[a-z]+(-[a-z]+)*$"
)  # e.g., "header-nav", "btn-primary"

TEXT_PLACEHOLDER = "..."
MAX_TEXT_LENGTH = 50
URL_THRESHOLD = 60  # Characters threshold to consider a URL for compression


def _is_random_class(class_name: str) -> bool:
    """Check if a class name appears to be randomly generated."""
    # If it looks purely semantic (lowercase letters and hyphens), keep it
    if SEMANTIC_CLASS_PATTERN.match(class_name):
        return False

    # If it contains digits, it's highly likely to be a hash/obfuscated
    if any(char.isdigit() for char in class_name):
        return True

    # Check specific known prefixes for obfuscation
    for pattern in RANDOM_CLASS_PATTERNS:
        if pattern.match(class_name):
            return True

    return False


@dataclass
class SkeletonConfig:
    """Configuration for HTML skeleton extraction."""

    keep_tags: frozenset[str] = field(default_factory=lambda: DEFAULT_KEEP_TAGS)
    remove_tags: frozenset[str] = field(default_factory=lambda: DEFAULT_REMOVE_TAGS)
    essential_attributes: frozenset[str] = field(
        default_factory=lambda: ESSENTIAL_ATTRIBUTES
    )
    text_placeholder: str = TEXT_PLACEHOLDER
    max_text_length: int = MAX_TEXT_LENGTH
    merge_single_children: bool = True
    remove_empty_elements: bool = True
    preserve_comments: bool = False
    min_word_count_for_block: int = 10
    compress_urls: bool = True  # New feature

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be at least 1")
        if self.min_word_count_for_block < 1:
            raise ValueError("min_word_count_for_block must be at least 1")
        if not self.keep_tags:
            raise ValueError("keep_tags cannot be empty")


class HTMLSkeletonExtractor:
    """
    Extracts skeletal structure from HTML for LLM context generation.
    Implements the block-tree approach described in HtmlRAG [[6]] with
    structural compression and semantic tag preservation.
    """

    def __init__(self, config: Optional[SkeletonConfig] = None):
        """
        Initialize the extractor with optional configuration.
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or SkeletonConfig()
        self.config.validate()
        self.url_map: Dict[str, str] = {}
        self.url_counter = 0

    def extract(self, html_content: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract skeletal structure from HTML content.
        Args:
            html_content: Raw HTML string
        Returns:
            Tuple of (Simplified HTML skeleton string, URL mapping dictionary)
        """
        self.url_map = {}
        self.url_counter = 0

        if not html_content or not html_content.strip():
            return "", {}

        soup = BeautifulSoup(html_content, "lxml")
        self._clean_document(soup)
        if self.config.merge_single_children:
            self._compress_structure(soup)
        if self.config.remove_empty_elements:
            self._remove_empty_elements(soup)

        skeleton = self._format_output(soup)
        return skeleton, self.url_map

    def _get_url_placeholder(self, url: str) -> str:
        """Generate a short placeholder for a URL and store the mapping."""
        if not self.config.compress_urls:
            return url

        if len(url) < URL_THRESHOLD:
            return url

        # Create a short hash-based ID
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        placeholder = f"u_{url_hash}"

        if placeholder not in self.url_map:
            self.url_map[placeholder] = url
            self.url_counter += 1

        return placeholder

    def _clean_document(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements and attributes from the document."""
        for tag_name in self.config.remove_tags:
            for element in soup.find_all(tag_name):
                element.decompose()

        if not self.config.preserve_comments:
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

        for tag in list(soup.find_all(True)):
            if tag.name not in self.config.keep_tags:
                if self._has_important_children(tag):
                    tag.unwrap()
                else:
                    tag.decompose()
                continue

            attrs_to_keep = {}
            for attr, value in list(tag.attrs.items()):
                if attr in self.config.essential_attributes:
                    if attr == "class":
                        classes = value if isinstance(value, list) else value.split()
                        filtered_classes = [
                            c for c in classes if not _is_random_class(c)
                        ]
                        if filtered_classes:
                            attrs_to_keep[attr] = filtered_classes[:3]
                    elif attr in ("href", "src"):
                        attrs_to_keep[attr] = self._get_url_placeholder(str(value))
                    else:
                        attrs_to_keep[attr] = value

            tag.attrs = attrs_to_keep
            self._truncate_text_content(tag)

    def _has_important_children(self, tag: Tag) -> bool:
        """Check if a tag contains children that should be preserved."""
        for child in tag.children:
            if isinstance(child, Tag):
                if child.name in self.config.keep_tags:
                    return True
                if self._has_important_children(child):
                    return True
            elif isinstance(child, NavigableString) and child.strip():
                return True
        return False

    def _truncate_text_content(self, tag: Tag) -> None:
        """Truncate or replace text content in a tag."""
        for child in list(tag.children):
            if isinstance(child, NavigableString) and not isinstance(child, Comment):
                text = str(child).strip()
                if text:
                    words = text.split()
                    if len(words) > self.config.max_text_length:
                        new_text = self.config.text_placeholder
                    elif len(text) > self.config.max_text_length * 5:
                        new_text = (
                            text[: self.config.max_text_length]
                            + self.config.text_placeholder
                        )
                    else:
                        new_text = text
                    new_string = NavigableString(new_text)
                    child.replace_with(new_string)

    def _compress_structure(self, soup: BeautifulSoup) -> None:
        """Merge single-child container elements to reduce nesting."""
        changed = True
        iterations = 0
        max_iterations = 10
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for tag in list(soup.find_all(True)):
                tag_children = [
                    child for child in tag.children if isinstance(child, Tag)
                ]
                if len(tag_children) == 1:
                    child = tag_children[0]
                    if tag.name in (
                        "div",
                        "span",
                        "section",
                        "article",
                    ) and child.name in ("div", "span", "section", "article"):
                        for attr, value in child.attrs.items():
                            if attr not in tag.attrs:
                                tag.attrs[attr] = value
                        child.unwrap()
                        changed = True
                        break

    def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove empty elements that don't contribute to structure."""
        changed = True
        iterations = 0
        max_iterations = 10
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for tag in list(soup.find_all(True)):
                has_text = bool(tag.get_text(strip=True))
                has_tag_children = any(isinstance(child, Tag) for child in tag.children)
                if not has_text and not has_tag_children:
                    if tag.name not in ("br", "hr", "img", "input"):
                        tag.decompose()
                        changed = True
                        break

    def _format_output(self, soup: BeautifulSoup) -> str:
        """Format the skeleton for output."""
        pretty_html = soup.prettify()
        lines = pretty_html.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.rstrip()
            if stripped or cleaned_lines:
                cleaned_lines.append(stripped)
        result_lines = []
        prev_blank = False
        for line in cleaned_lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            result_lines.append(line)
            prev_blank = is_blank
        return "\n".join(result_lines)


class BlockTreeBuilder:
    """
    Builds a block tree from HTML for hierarchical analysis.
    Implements the granularity-adjustable block tree construction from HtmlRAG [[6]].
    """

    def __init__(self, max_words_per_block: int = 50):
        """
        Initialize block tree builder.
        Args:
            max_words_per_block: Maximum words per block for granularity control
        """
        self.max_words_per_block = max_words_per_block

    def build_block_tree(self, html_content: str) -> dict:
        """
        Build a block tree from HTML content.
        Args:
            html_content: Raw HTML string
        Returns:
            Nested dictionary representing the block tree
        """
        if not html_content or not html_content.strip():
            return {}
        soup = BeautifulSoup(html_content, "lxml")
        root_element = soup.body or soup
        root = self._build_node(root_element)
        return root or {}

    def _build_node(self, element) -> Optional[dict]:
        """Recursively build a block tree node."""
        node = {
            "tag": element.name if isinstance(element, Tag) else "#text",
            "attributes": {},
            "children": [],
            "text_content": "",
            "word_count": 0,
        }
        if isinstance(element, Tag):
            for attr in ESSENTIAL_ATTRIBUTES:
                if attr in element.attrs:
                    value = element.attrs[attr]
                    if isinstance(value, list):
                        node["attributes"][attr] = " ".join(str(v) for v in value[:3])
                    else:
                        node["attributes"][attr] = str(value)[:100]
            for child in element.children:
                child_node = self._build_node(child)
                if child_node:
                    node["children"].append(child_node)
            text = element.get_text(separator=" ", strip=True)
            node["text_content"] = text[:200]
            node["word_count"] = len(text.split())
        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                node["text_content"] = text[:200]
                node["word_count"] = len(text.split())
            else:
                return None
        return node

    def prune_by_relevance(self, block_tree: dict, query: str) -> dict:
        """
        Prune block tree based on relevance to query.
        Args:
            block_tree: The block tree structure
            query: Search query for relevance calculation
        Returns:
            Pruned block tree
        """
        if not block_tree or not query:
            return block_tree
        query_words = set(query.lower().split())
        pruned = self._prune_node(block_tree, query_words)
        return pruned or {}

    def _prune_node(self, node: dict, query_words: set) -> Optional[dict]:
        """Recursively prune nodes based on relevance."""
        if not node:
            return None
        text_lower = node.get("text_content", "").lower()
        text_words = set(text_lower.split())
        relevance = len(query_words & text_words) / max(len(query_words), 1)
        pruned_children = []
        for child in node.get("children", []):
            pruned_child = self._prune_node(child, query_words)
            if pruned_child:
                pruned_children.append(pruned_child)
        if relevance > 0.3 or pruned_children:
            node["children"] = pruned_children
            node["relevance_score"] = round(relevance, 3)
            return node
        return None


def extract_skeleton(
    html_content: str, config: Optional[SkeletonConfig] = None
) -> Tuple[str, Dict[str, str]]:
    """
    Convenience function to extract HTML skeleton.
    Args:
        html_content: Raw HTML string
        config: Optional configuration
    Returns:
        Tuple of (Simplified HTML skeleton string, URL mapping dictionary)
    """
    extractor = HTMLSkeletonExtractor(config)
    return extractor.extract(html_content)


def build_block_tree(html_content: str, max_words: int = 50) -> dict:
    """
    Convenience function to build block tree.
    Args:
        html_content: Raw HTML string
        max_words: Maximum words per block
    Returns:
        Block tree dictionary
    """
    builder = BlockTreeBuilder(max_words_per_block=max_words)
    return builder.build_block_tree(html_content)

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# Configuration constants
DEFAULT_KEEP_TAGS = frozenset(
    [
        # Document structure
        "html",
        "head",
        "body",
        # Semantic sections
        "header",
        "footer",
        "nav",
        "main",
        "section",
        "article",
        "aside",
        # Content structure
        "div",
        "p",
        "span",
        "br",
        "hr",
        # Headings
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        # Lists
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        # Tables
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "caption",
        # Forms
        "form",
        "input",
        "button",
        "select",
        "option",
        "textarea",
        "label",
        "fieldset",
        "legend",
        # Links and media
        "a",
        "img",
        "figure",
        "figcaption",
        "picture",
        "source",
        # Other semantic elements
        "strong",
        "em",
        "b",
        "i",
        "u",
        "code",
        "pre",
        "blockquote",
        "cite",
        # Interactive
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

TEXT_PLACEHOLDER = "..."
MAX_TEXT_LENGTH = 50


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

    def extract(self, html_content: str) -> str:
        """
        Extract skeletal structure from HTML content.

        Args:
            html_content: Raw HTML string

        Returns:
            Simplified HTML skeleton string
        """
        if not html_content or not html_content.strip():
            return ""

        # Parse HTML
        soup = BeautifulSoup(html_content, "lxml")

        # Clean the document
        self._clean_document(soup)

        # Compress structure
        if self.config.merge_single_children:
            self._compress_structure(soup)

        if self.config.remove_empty_elements:
            self._remove_empty_elements(soup)

        # Format output
        return self._format_output(soup)

    def _clean_document(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements and attributes from the document."""

        # Remove unwanted tags completely
        for tag_name in self.config.remove_tags:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove comments if not preserving them
        if not self.config.preserve_comments:
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

        # Process remaining tags
        for tag in list(soup.find_all(True)):
            # Skip tags not in keep list
            if tag.name not in self.config.keep_tags:
                # Check if it has important children before removing
                if self._has_important_children(tag):
                    tag.unwrap()  # Keep children, remove tag
                else:
                    tag.decompose()  # Remove tag and children
                continue

            # Filter attributes
            attrs_to_keep = {}
            for attr, value in list(tag.attrs.items()):
                if attr in self.config.essential_attributes:
                    attrs_to_keep[attr] = value

            # Special handling for class - keep but truncate if too long
            if "class" in attrs_to_keep:
                classes = attrs_to_keep["class"]
                if isinstance(classes, list):
                    # Keep only first few meaningful classes
                    attrs_to_keep["class"] = classes[:3]

            tag.attrs = attrs_to_keep

            # Truncate text content
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
                        # Replace with placeholder
                        new_text = self.config.text_placeholder
                    elif len(text) > self.config.max_text_length * 5:  # Character limit
                        new_text = (
                            text[: self.config.max_text_length]
                            + self.config.text_placeholder
                        )
                    else:
                        new_text = text

                    # Replace the text node
                    new_string = NavigableString(new_text)
                    child.replace_with(new_string)

    def _compress_structure(self, soup: BeautifulSoup) -> None:
        """Merge single-child container elements to reduce nesting."""
        changed = True
        iterations = 0
        max_iterations = 10  # Prevent infinite loops

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for tag in list(soup.find_all(True)):
                # Get direct children that are tags
                tag_children = [
                    child for child in tag.children if isinstance(child, Tag)
                ]

                if len(tag_children) == 1:
                    child = tag_children[0]

                    # Merge if both parent and child are generic containers
                    if tag.name in (
                        "div",
                        "span",
                        "section",
                        "article",
                    ) and child.name in ("div", "span", "section", "article"):
                        # Move child's attributes to parent if parent doesn't have them
                        for attr, value in child.attrs.items():
                            if attr not in tag.attrs:
                                tag.attrs[attr] = value

                        # Unwrap the child (move its children to parent)
                        child.unwrap()
                        changed = True
                        break  # Restart iteration after modification

    def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove empty elements that don't contribute to structure."""
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for tag in list(soup.find_all(True)):
                # Check if tag is empty (no text, no children)
                has_text = bool(tag.get_text(strip=True))
                has_tag_children = any(isinstance(child, Tag) for child in tag.children)

                if not has_text and not has_tag_children:
                    # Don't remove certain structural elements even if empty
                    if tag.name not in ("br", "hr", "img", "input"):
                        tag.decompose()
                        changed = True
                        break  # Restart iteration

    def _format_output(self, soup: BeautifulSoup) -> str:
        """Format the skeleton for output."""
        # Get prettified HTML
        pretty_html = soup.prettify()

        # Clean up excessive whitespace
        lines = pretty_html.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.rstrip()
            if stripped or cleaned_lines:  # Keep non-empty lines or maintain structure
                cleaned_lines.append(stripped)

        # Remove excessive blank lines
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
            # Extract essential attributes
            for attr in ESSENTIAL_ATTRIBUTES:
                if attr in element.attrs:
                    value = element.attrs[attr]
                    if isinstance(value, list):
                        node["attributes"][attr] = " ".join(str(v) for v in value[:3])
                    else:
                        node["attributes"][attr] = str(value)[:100]

            # Process children
            for child in element.children:
                child_node = self._build_node(child)
                if child_node:
                    node["children"].append(child_node)

            # Calculate word count
            text = element.get_text(separator=" ", strip=True)
            node["text_content"] = text[:200]  # Truncate for display
            node["word_count"] = len(text.split())

        elif isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                node["text_content"] = text[:200]
                node["word_count"] = len(text.split())
            else:
                return None  # Skip empty text nodes

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

        # Simple keyword-based relevance scoring
        query_words = set(query.lower().split())

        pruned = self._prune_node(block_tree, query_words)
        return pruned or {}

    def _prune_node(self, node: dict, query_words: set) -> Optional[dict]:
        """Recursively prune nodes based on relevance."""
        if not node:
            return None

        # Calculate relevance score
        text_lower = node.get("text_content", "").lower()
        text_words = set(text_lower.split())
        relevance = len(query_words & text_words) / max(len(query_words), 1)

        # Prune children first
        pruned_children = []
        for child in node.get("children", []):
            pruned_child = self._prune_node(child, query_words)
            if pruned_child:
                pruned_children.append(pruned_child)

        # Keep node if it has high relevance or has relevant children
        if relevance > 0.3 or pruned_children:
            node["children"] = pruned_children
            node["relevance_score"] = round(relevance, 3)
            return node

        return None


def extract_skeleton(html_content: str, config: Optional[SkeletonConfig] = None) -> str:
    """
    Convenience function to extract HTML skeleton.

    Args:
        html_content: Raw HTML string
        config: Optional configuration

    Returns:
        Simplified HTML skeleton string
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

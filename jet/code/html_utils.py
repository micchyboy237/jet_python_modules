import re
import justext
from typing import List, Optional
from bs4 import BeautifulSoup
from lxml import etree, html

from jet.logger import logger
from jet.utils.text import fix_and_unidecode
from jet.transformers.formatters import format_html
import justext.paragraph


def is_html(text: str) -> bool:
    """Determines if the input string contains HTML-like content.

    This function checks for the presence of HTML tags or structure without requiring
    strict HTML validity. It returns True for strings containing HTML tags, even if
    incomplete or malformed, and False for plain text, Markdown, or non-HTML content.

    Args:
        text: The input string to check.

    Returns:
        True if the input contains HTML-like content, False otherwise.
    """
    # Handle empty or whitespace-only input
    if not text or not text.strip():
        logger.debug("Input is empty or whitespace-only")
        return False

    try:
        # Normalize input by stripping leading/trailing whitespace
        text = text.strip()

        # Check for common HTML indicators
        html_indicators = [
            r'<\w+[\s>]',  # Opening tag like <tag> or <tag attr>
            r'</\w+>',     # Closing tag like </tag>
            r'<!DOCTYPE\s+html\s*>',  # DOCTYPE declaration
            r'<!--.*?-->',  # HTML comments
        ]

        # Quick regex check for HTML-like patterns
        for pattern in html_indicators:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                logger.debug(f"HTML pattern found: {pattern}")
                return True

        # Use BeautifulSoup to parse and check for tags
        soup = BeautifulSoup(text, 'html.parser')
        # Check if the parsed content has any tags (excluding text-only content)
        has_tags = any(tag.name for tag in soup.find_all())

        if has_tags:
            logger.debug("HTML tags detected by BeautifulSoup")
            return True

        logger.debug("No HTML tags or patterns detected")
        return False
    except Exception as e:
        logger.debug(f"Error checking HTML: {str(e)}")
        return False


def valid_html(text: str) -> bool:
    """Validates if the input string is valid HTML, rejecting Markdown and non-HTML content.

    Args:
        text: The input string to validate.

    Returns:
        True if the input is valid HTML, False otherwise.
    """
    # Check for empty or whitespace-only input
    if not text or not text.strip():
        return False

    try:
        # Normalize input by stripping leading/trailing whitespace
        text = text.strip()

        # Check if input starts with <!DOCTYPE html> (case-insensitive)
        doctype_pattern = r'^\s*<!DOCTYPE\s+html\s*>'
        is_doctype = re.match(doctype_pattern, text, re.IGNORECASE) is not None

        # Prepare text for parsing
        parse_text = text
        if is_doctype:
            # Remove DOCTYPE declaration
            parse_text = re.sub(doctype_pattern, '', text,
                                count=1, flags=re.IGNORECASE).strip()

        # Check if the input is a complete HTML document (starts with <html>)
        is_complete_html = re.match(
            r'^\s*<html\b', parse_text, re.IGNORECASE) is not None
        if not is_complete_html:
            # Wrap fragments in <html><body>...</body></html>
            parse_text = f"<html><body>{parse_text}</body></html>"

        # Use XMLParser for strict validation
        parser = etree.XMLParser(
            recover=False, no_network=True, remove_comments=True)
        doc = etree.fromstring(parse_text, parser=parser)

        # Find body (for fragments) or root (for complete documents)
        body = doc if is_complete_html else doc.find("body")
        if body is None:
            return False

        # Check for meaningful HTML content (elements, not just text)
        has_elements = any(child.tag is not etree.Comment for child in body)

        # Reject if body contains only text and no elements
        if not has_elements and body.text and body.text.strip():
            return False

        # Check for Markdown-like patterns in text nodes
        markdown_patterns = [
            r'^\s*#+\s',  # Headings (#, ##, etc.)
            r'\*\*',      # Bold (**text**)
            r'^\s*-\s',   # Lists (- item)
            r'^\s*\|\s',  # Tables (| ... |)
            r'\[.*?\]\(.*?\)',  # Links ([text](url))
            r'!\[.*?\]\(.*?\)',  # Images (![text](url))
            r'^\s*>\s',   # Blockquotes (> text)
            r'\[\^.*?\]',  # Footnotes ([^1])
        ]

        def check_for_markdown(node):
            # Check text and tail of the node for Markdown patterns
            for text in [node.text, node.tail]:
                if text and text.strip():
                    for pattern in markdown_patterns:
                        if re.search(pattern, text, re.MULTILINE):
                            return True
            # Recursively check child nodes
            for child in node:
                if check_for_markdown(child):
                    return True
            return False

        if check_for_markdown(body):
            return False

        return True
    except etree.ParseError:
        return False


def remove_html_comments(text: str) -> str:
    """Remove all HTML comments from a string, including multiline ones.

    Args:
        text: The input HTML or text string.

    Returns:
        The text with HTML comments removed.
    """
    logger.debug(f"Input text: {text}")

    # Handle empty or whitespace-only input
    if not text or not text.strip():
        logger.debug("Input is empty or whitespace-only")
        return ''

    try:
        # Check if input is a complete HTML document
        doctype_pattern = r'^\s*<!DOCTYPE\s+html\s*>'
        is_doctype = re.match(doctype_pattern, text, re.IGNORECASE) is not None
        parse_text = text
        if is_doctype:
            parse_text = re.sub(doctype_pattern, '', text,
                                count=1, flags=re.IGNORECASE).strip()

        is_complete_html = re.match(
            r'^\s*<html\b', parse_text, re.IGNORECASE) is not None
        if not is_complete_html:
            # Wrap fragments in <html><body>...</body></html>
            parse_text = f"<html><body>{parse_text}</body></html>"
            logger.debug(f"Wrapped text: {parse_text}")

        # Parse with HTMLParser, allowing recovery for malformed inputs
        parser = etree.HTMLParser(recover=True, no_network=True)
        tree = etree.fromstring(parse_text, parser=parser)

        # Remove all comment nodes
        for comment in tree.xpath("//comment()"):
            comment.getparent().remove(comment)

        # Serialize back to string
        if is_complete_html:
            # For complete HTML, serialize the entire tree
            result = etree.tostring(
                tree, encoding='unicode', method='html').strip()
        else:
            # For fragments, serialize the body content
            body = tree.find("body")
            if body is None:
                logger.debug("No body element found")
                return ''
            # Serialize all children of body, including text and tails
            result = ''
            if body.text and body.text.strip():
                result += body.text
            for child in body:
                # Serialize the child element, including its tail
                child_str = etree.tostring(
                    child, encoding='unicode', method='html').strip()
                result += child_str
                if child.tail and child.tail.strip():
                    result += child.tail

        logger.debug(f"Result after removing comments: {result}")
        return result if result else ''
    except etree.ParseError as e:
        logger.debug(f"Parse error: {str(e)}")
        return text  # Return original text on parsing failure


def preprocess_html(html: str) -> str:
    """
    Preprocess HTML content by removing unwanted elements, comments, adding spacing,
    transforming dt/dd pairs (with optional div wrappers) to li tags within ul,
    and inserting the title as an h1 at the beginning of the body if no h1 is the first child.

    Args:
        html: Input HTML string to preprocess.

    Returns:
        Preprocessed HTML string.
    """
    # Extract the title from the <title> tag
    title_match = re.search(
        r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Default Title"

    # Check if the first child of <body> is an <h1>
    body_match = re.search(r'(<body[^>]*>)\s*(<[^>]+>)', html, re.IGNORECASE)
    has_h1_first = False
    if body_match:
        first_child = body_match.group(2)
        has_h1_first = bool(
            re.match(r'<h1(?:\s+[^>]*)?>', first_child, re.IGNORECASE))

    # Remove unwanted elements (button, script, style, form, input, select, textarea)
    unwanted_elements = r'button|script|style|form|input|select|textarea'
    pattern_unwanted = rf'<({unwanted_elements})(?:\s+[^>]*)?>.*?</\1>'
    html = re.sub(pattern_unwanted, '', html, flags=re.DOTALL)

    # Remove HTML comments
    html = re.sub(r'<!--[\s\S]*?-->', '', html, flags=re.DOTALL)

    # Convert <dt> and <dd> pairs to li tags, removing optional <div> wrappers
    def replace_dt_dd(match):
        dt_content = match.group(2).strip() if match.group(
            2) else match.group(5).strip()
        dd_content = match.group(3).strip() if match.group(
            3) else match.group(6).strip()
        return f'<li>{dd_content}: {dt_content}</li>'

    # Replace <dt> and <dd> pairs, optionally wrapped in <div>
    html = re.sub(
        r'(<div[^>]*>\s*)?<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>(\s*</div>)?',
        replace_dt_dd,
        html,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Convert <dl> to <ul>
    html = re.sub(r'<dl[^>]*>', '<ul>', html, flags=re.IGNORECASE)
    html = re.sub(r'</dl>', '</ul>', html, flags=re.IGNORECASE)

    # Add space between consecutive inline elements
    inline_elements = r'span|a|strong|em|b|i|code|small|sub|sup|mark|del|ins|q'
    pattern_inline = rf'</({inline_elements})><({inline_elements})'
    html = re.sub(pattern_inline, r'</\1> <\2', html)

    # Insert title as <h1> right after <body> tag only if no <h1> is the first child
    if not has_h1_first:
        html = re.sub(
            r'(<body[^>]*>)', rf'\1<h1>{title}</h1>', html, flags=re.IGNORECASE)

    html = fix_and_unidecode(html)

    return html


def clean_html(html: str, max_link_density: float = 0.2, max_link_ratio: float = 0.3, language: str = "English") -> List[justext.paragraph.Paragraph]:
    paragraphs = justext.justext(
        html,
        justext.get_stoplist(language),
        max_link_density=max_link_density,
        length_low=50,
        length_high=150,
        no_headings=False
    )
    filtered_paragraphs = [
        p for p in paragraphs
        if (p.is_heading or (not p.is_boilerplate and p.links_density() < max_link_ratio))
    ]
    return filtered_paragraphs

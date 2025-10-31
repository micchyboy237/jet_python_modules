import re
from typing import List, Tuple
from bs4 import BeautifulSoup
from lxml import etree

from jet.logger import logger
from jet.utils.code_utils import normalize_multiline_spaces
from jet.utils.text import fix_and_unidecode


def is_html(text: str) -> bool:
    """
    Detects whether the input text is likely to contain HTML content.

    Args:
        text (str): Input text to check.

    Returns:
        bool: True if HTML is detected, False otherwise.
    """
    if not text or not text.strip():
        # Empty or whitespace-only input
        return False

    stripped = text.strip()

    # Quick pattern-based checks for clear HTML markers
    html_patterns = [
        r"<!DOCTYPE\s+html>",  # DOCTYPE
        r"<!--.*?-->",         # HTML comment
        r"<[a-zA-Z]+[^>]*>",   # Opening tag
        r"</[a-zA-Z]+>",       # Closing tag
        r"<[a-zA-Z]+[^>]*?$",  # Incomplete tag (no closing >)
    ]
    for pattern in html_patterns:
        if re.search(pattern, stripped, flags=re.IGNORECASE | re.DOTALL):
            return True

    # Fallback: Use BeautifulSoup to detect if parsing changes the text
    soup = BeautifulSoup(stripped, "html.parser")

    # If BeautifulSoup finds any tags or special structures, it's HTML
    if soup.find() or soup.find_all(True):
        return True

    # If parsing didn't create any tags and output is identical, not HTML
    parsed_text = soup.get_text(strip=True)
    if parsed_text == stripped:
        return False

    # Edge fallback: Sometimes malformed HTML still alters parsing
    return bool(soup.contents)


def valid_html(text: str) -> bool:
    """Validates if the input string is valid HTML string.

    Args:
        text: The input string to validate.

    Returns:
        True if the input is valid HTML, False otherwise.
    """
    return is_html(text)


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


def preprocess_html(html: str, includes: list[str] | None = None, excludes: list[str] | None = None) -> str:
    """
    Preprocess HTML content by:
      - Removing unwanted elements, comments.
      - Optionally filtering by `includes` or `excludes`.
      - Adding spacing between inline elements.
      - Transforming <dt>/<dd> pairs into <li> within <ul>.
      - Inserting title as <h1> if not already first element.
    """
    includes = set(includes or [])
    excludes = set(excludes or [])

    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Default Title"

    # Remove unwanted elements
    unwanted_elements = r'button|script|style|form|input|select|textarea'
    html = re.sub(rf'<({unwanted_elements})(?:\s+[^>]*)?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove comments
    html = re.sub(r'<!--[\s\S]*?-->', '', html, flags=re.DOTALL)

    # Apply includes/excludes filtering (after unwanted removal)
    if includes:
        # Keep only specified tags and their content
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(True):
            if tag.name not in includes and tag.name not in {"html", "head", "title", "body", "h1"}:
                tag.decompose()
        html = str(soup)
    elif excludes:
        # Remove specific tags
        soup = BeautifulSoup(html, "html.parser")
        for tag_name in excludes:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        html = str(soup)

    # Convert <dl><dt><dd> sections into Markdown definition lists
    html = convert_dl_blocks_to_md(html)

    # # Add spaces between inline elements
    # inline_elements = r'span|a|strong|em|b|i|code|small|sub|sup|mark|del|ins|q'
    # html = re.sub(rf'</({inline_elements})><({inline_elements})', r'</\1> <\2', html)

    # # Add <h1> if not already first
    # body_match = re.search(r'(<body[^>]*>)\s*(<[^>]+>)', html, re.IGNORECASE)
    # has_h1_first = False
    # if body_match:
    #     first_child = body_match.group(2)
    #     has_h1_first = bool(re.match(r'<h1(?:\s+[^>]*)?>', first_child, re.IGNORECASE))
    # if not has_h1_first:
    #     html = re.sub(r'(<body[^>]*>)', rf'\1<h1>{title}</h1>', html, flags=re.IGNORECASE)

    html = fix_and_unidecode(html)
    return html


def dl_to_md(match: re.Match) -> str:
    """
    Convert a <dl>...</dl> HTML block to Markdown definition list syntax.
    Example output:

    Term One
    : Definition one.

    Term Two
    : Definition A
    : Definition B
    """
    inner_html = match.group(1)
    soup = BeautifulSoup(inner_html, "html.parser")

    items: List[Tuple[str, List[str]]] = []
    current_term: str | None = None

    for node in soup.find_all(["dt", "dd"]):
        text = " ".join(node.stripped_strings)
        text = re.sub(r"\s+", " ", text).strip()

        if node.name.lower() == "dt":
            current_term = text
            items.append((current_term, []))
        elif node.name.lower() == "dd":
            if not items:
                items.append(("", [text]))
                current_term = ""
            else:
                items[-1][1].append(text)

    out_lines: List[str] = []
    for term, defs in items:
        if term:
            out_lines.append(term)
        for d in defs:
            out_lines.append(f": {d}")
        out_lines.append("")  # blank line between groups

    md = "\n".join(out_lines).rstrip() + "\n\n"
    return md


def convert_dl_blocks_to_md(html: str) -> str:
    """Replace all <dl>...</dl> blocks in HTML with Markdown definition lists wrapped in <pre> to preserve newlines."""
    def repl(match: re.Match) -> str:
        md_content = dl_to_md(match)
        return f"<pre class=\"jet-dl-block\">{md_content}</pre>"

    result = re.sub(
        r"<dl[^>]*>\s*(.*?)\s*</dl>",
        repl,
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return normalize_multiline_spaces(result)


def clean_html(html: str, max_link_density: float = 0.2, max_link_ratio: float = 0.3, language: str = "English"): # -> List[justext.paragraph.Paragraph]:
    import justext
    import justext.paragraph
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

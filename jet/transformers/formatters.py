import json
import re
from typing import Optional
from bs4 import BeautifulSoup, NavigableString, Tag
from jet.transformers.object import make_serializable


def prettify_value(prompt, level=0):
    """
    Recursively builds a formatted log string from a nested dictionary or list.

    :param prompt: Dictionary or list to process.
    :param level: Indentation level for nested structures.
    :return: Formatted string for the log.
    """
    prompt_log = ""
    indent = " " * level  # Indentation for nested structures
    marker_list = ["-", "+"]
    marker = marker_list[level % 2]
    line_prefix = indent if level == 0 else f"{indent} {marker} "

    if isinstance(prompt, dict):
        for key, value in prompt.items():
            capitalized_key = key.capitalize()
            if isinstance(value, (dict, list)):  # If nested structure
                prompt_log += f"{line_prefix}{capitalized_key}:\n"
                prompt_log += prettify_value(value, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{capitalized_key}: {value}\n"
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, (dict, list)):  # If nested structure
                prompt_log += prettify_value(item, level + 1)
            else:  # Primitive value
                prompt_log += f"{line_prefix}{item}\n"

    return prompt_log


def format_json(value, indent: Optional[int] = 2):
    serialized = make_serializable(value)
    return json.dumps(serialized, indent=indent)


def minify_html(html: str) -> str:
    # Remove newlines and tabs
    html = re.sub(r'\s*\n\s*', '', html)
    html = re.sub(r'\s*\t\s*', '', html)
    # Remove spaces between tags
    html = re.sub(r'>\s+<', '><', html)
    return html


def format_html(html: str, indent: int = 2) -> str:
    """
    Beautifies an HTML string by correctly indenting each level.
    Keeps void elements' start and end tags inline and preserves spaces in text-containing elements.

    Args:
        html (str): The input HTML string to beautify
        indent (int): Number of spaces for each indentation level (default: 2)

    Returns:
        str: The beautified HTML string with proper indentation
    """
    # Void elements that cannot have children
    VOID_ELEMENTS = {
        'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
        'link', 'meta', 'param', 'source', 'track', 'wbr'
    }

    # Elements that inherently contain text
    TEXT_ELEMENTS = {
        'a', 'abbr', 'b', 'bdi', 'bdo', 'cite', 'code', 'del', 'dfn',
        'em', 'i', 'ins', 'kbd', 'mark', 'q', 's', 'samp', 'small',
        'span', 'strong', 'sub', 'sup', 'time', 'u', 'var', 'p',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label', 'legend',
        'option', 'title', 'figcaption'
    }

    html = minify_html(html)

    # Normalize input HTML
    html = ' '.join(html.split()).replace('> <', '><').strip()

    result = []
    current_indent = 0
    i = 0

    while i < len(html):
        if html[i] == '<':
            # Extract tag content
            end = html.find('>', i)
            tag_content = html[i + 1:end]
            is_closing = tag_content.startswith('/')
            tag_name = tag_content.lstrip('/').split()[0].lower()

            # Handle closing tags (not for void elements, as they are handled inline)
            if is_closing and tag_name not in VOID_ELEMENTS:
                current_indent -= 1
                tag = html[i:end + 1]
                result.append(' ' * (current_indent * indent) + tag)
                i = end + 1
            # Handle void elements
            elif tag_name in VOID_ELEMENTS:
                # Check if the next segment is the closing tag
                next_start = html.find('<', end + 1)
                if next_start != -1 and html[end + 1:next_start].strip() == '':
                    next_tag = html[next_start:html.find('>', next_start) + 1]
                    if next_tag == f'</{tag_name}>':
                        # Combine start and end tags inline
                        tag = html[i:html.find('>', next_start) + 1]
                        result.append(' ' * (current_indent * indent) + tag)
                        i = html.find('>', next_start) + 1
                    else:
                        # Just the opening tag (self-closing or malformed)
                        tag = html[i:end + 1]
                        result.append(' ' * (current_indent * indent) + tag)
                        i = end + 1
                else:
                    # Just the opening tag
                    tag = html[i:end + 1]
                    result.append(' ' * (current_indent * indent) + tag)
                    i = end + 1
            # Handle comments
            elif tag_content.startswith('!'):
                tag = html[i:end + 1]
                result.append(' ' * (current_indent * indent) + tag)
                i = end + 1
            # Handle opening tags
            else:
                tag = html[i:end + 1]
                result.append(' ' * (current_indent * indent) + tag)
                if tag_name not in TEXT_ELEMENTS:
                    current_indent += 1
                i = end + 1
        else:
            # Handle text content
            end = html.find('<', i)
            if end == -1:
                end = len(html)
            text = html[i:end]
            # Only strip text if not within text element context
            if result and result[-1].strip().startswith('<') and result[-1].strip()[1:].split()[0].lower() not in TEXT_ELEMENTS:
                text = text.strip()
            if text:
                result.append(' ' * (current_indent * indent) + text)
            i = end

    return '\n'.join(line.rstrip() for line in result if line.strip())


__all__ = [
    "format_json",
    "minify_html",
    "format_html",
]

# Example Usage
if __name__ == "__main__":
    # Example for prettify_value
    prompt = {
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {
                "email": "alice@example.com",
                "phone": "123-456-7890"
            }
        },
        "status": "active"
    }
    prompt_log = prettify_value(prompt)
    print("Formatted Prompt:")
    print(prompt_log)

    # Example for format_json
    json_output = format_json(prompt, indent=4)
    print("\nFormatted JSON:")
    print(json_output)

    # Example for format_html
    html_input = '<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello <b>World</b></p></body></html>'
    formatted_html = format_html(html_input)
    print("\nFormatted HTML:")
    print(formatted_html)

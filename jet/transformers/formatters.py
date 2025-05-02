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


def format_html(html: str, indent: Optional[int] = 2) -> str:
    """
    Formats an HTML string with proper indentation and line breaks, ensuring all nested
    elements are indented correctly according to their depth.

    :param html: HTML string to format.
    :param indent: Number of spaces for each indentation level (default: 2).
    :return: Formatted HTML string.
    """
    # Handle empty or whitespace-only input
    if not html or html.isspace():
        return ""

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    def format_element(element, level: int = 0) -> list:
        """
        Recursively formats an element and its children, applying correct indentation.

        :param element: BeautifulSoup Tag or NavigableString.
        :param level: Current indentation level.
        :return: List of formatted lines.
        """
        lines = []

        if isinstance(element, NavigableString):
            # Handle text nodes
            text = str(element).strip()
            if text:
                lines.append(' ' * (level * indent) + text)
            return lines

        if isinstance(element, Tag):
            # Opening tag
            tag_str = f"<{element.name}"
            if element.attrs:
                for attr, value in element.attrs.items():
                    tag_str += f' {attr}="{value}"'
            tag_str += ">"
            lines.append(' ' * (level * indent) + tag_str)

            # Process children
            for child in element.children:
                child_lines = format_element(child, level + 1)
                lines.extend(child_lines)

            # Closing tag
            lines.append(' ' * (level * indent) + f"</{element.name}>")

        return lines

    # Start formatting from the root
    formatted_lines = []
    if soup.html:
        # Handle DOCTYPE if present
        if soup.find(string=lambda s: isinstance(s, str) and s.strip().startswith('<!DOCTYPE')):
            formatted_lines.append('<!DOCTYPE html>')

        # Format the <html> tag and its children
        formatted_lines.extend(format_element(soup.html, 0))
    else:
        # If no <html> tag, format all top-level elements
        for element in soup.children:
            formatted_lines.extend(format_element(element, 0))

    # Join lines
    formatted = '\n'.join(line.rstrip()
                          for line in formatted_lines if line.strip())

    return formatted.rstrip()


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
    formatted_html = format_html(html_input, indent=4)
    print("\nFormatted HTML:")
    print(formatted_html)

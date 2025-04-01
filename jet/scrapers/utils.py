from collections import defaultdict
from typing import List, Optional
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

from typing import List


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


class TreeNode(Dict):
    tag: str
    text: Optional[str]  # Some nodes may not contain text
    depth: int
    id: Optional[str]  # ID attribute of the element
    class_names: List[str]  # List of class names
    children: List['TreeNode']  # Recursive reference to TreeNode


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


def extract_tree_with_text(html: str, excludes: List[str] = ["style", "script"]) -> Optional[TreeNode]:
    """
    Finds all elements that contain any text, ensuring a natural document order.
    Returns a tree-like structure of parents and their corresponding text, including depth for each node.
    Only includes text for nodes that directly hold it.

    :param html: The HTML string to parse.
    :return: A tree-like structure with parent elements and their corresponding text.
    """
    # Helper function to recursively build the tree
    def build_tree(element, current_depth: int) -> Optional[TreeNode]:
        text = pq(element).text().strip()

        # Extract ID and class name (only if exists)
        element_id = pq(element).attr('id')
        element_class = pq(element).attr('class')

        # Define shared regex pattern for valid HTML identifiers
        valid_id_pattern = r'^[a-zA-Z_-]+$'
        # Filter non-alphabet id and class names
        id = element_id if element_id and re.match(
            valid_id_pattern, element_id) else None
        # Split class names into a list if they exist
        class_names = [name for name in (element_class.split() if element_class else [])
                       if re.match(valid_id_pattern, name)]

        # Include text only for leaf nodes that directly hold text
        if text and len(pq(element).children()) == 0:  # No children, direct text
            return {
                "tag": pq(element)[0].tag,
                "text": text,
                "depth": current_depth,
                "id": element_id,
                "class_names": class_names,  # Store class names as a list
                "children": []  # No children in this case
            }

        # Otherwise, process children recursively
        children = []
        for child in pq(element).children():
            child_tree = build_tree(child, current_depth + 1)
            if child_tree:
                children.append(child_tree)

        # Return the element if it has children containing text
        if children:
            return {
                "tag": pq(element)[0].tag,
                "text": None,  # No text for container elements
                "depth": current_depth,
                "id": id,
                "class_names": class_names,  # Store class names as a list
                "children": children
            }
        return None

    doc = pq(html)
    # Apply the exclusion logic before building the tree
    exclude_elements(doc, excludes)
    # Start with the root element (<html>) at depth 0
    root = build_tree(doc[0], 0)

    return root  # Returns tree-like structure starting from <html> element


# Function to print the tree-like structure recursively
def print_tree(node: TreeNode, indent=0, excludes: List[str] = ["style", "script"]):
    if node:
        # Skip node if its tag is in the excludes list
        if node['tag'] in excludes:
            return

        # or node['children'][0]['text']:
        if node['text'] or node['id'] or node['class_names'] or node['children'][0]['text']:
            tag_text = node['tag']
            if node['id']:
                tag_text += " " + colorize_log(f"#{node['id']}", "YELLOW")
            if node['class_names']:
                tag_text += " " + \
                    colorize_log(
                        ', '.join([f".{class_name}" for class_name in node['class_names']]), "ORANGE")

            if node['text']:
                logger.log(('  ' * indent + f"{node['depth']}:"), tag_text, "-",
                           json.dumps(node['text'][:30]), colors=["INFO", "DEBUG", "GRAY", "SUCCESS"])
            else:
                logger.log(
                    ('  ' * indent + f"{node['depth']}:"), tag_text, colors=["INFO", "DEBUG"])

        for child in node['children']:
            print_tree(child, indent + 1, excludes)


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

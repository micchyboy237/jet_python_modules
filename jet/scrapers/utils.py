import re
import string
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
    text = unidecode.unidecode(text)

    # lemmas = lemmatize_text(text)
    # text = ' '.join(lemmas).strip()
    text = clean_newlines(text)
    # text = clean_spaces(text, exclude_chars=["-", "\n"])
    text = clean_non_ascii(text)
    text = clean_other_characters(text)

    return text.strip()


def clean_newlines(content) -> str:
    """Merge consecutive newlines from the content."""
    # Remove trailing whitespace for each line
    content = '\n'.join([line.rstrip() for line in content.split('\n')])
    # Reduce consecutive newlines to a single newline
    content = re.sub(r'\n+', '\n', content)
    return content


def clean_spaces(content: str, exclude_chars: List[str] = []) -> str:
    """Removes consecutive spaces and ensures proper spacing around punctuation, 
    with an option to exclude specific punctuation from spacing adjustments."""

    # Create a regex pattern for excluded punctuation characters
    punctuation_pattern = f"[{re.escape(string.punctuation)}]"

    # Ensure single spacing around punctuation except for excluded characters
    def spacing_match(m):
        char = m.group(1)
        return char if char in exclude_chars else f"{char} "

    content = re.sub(f'\s*({punctuation_pattern})\s*', spacing_match, content)

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


__all__ = [
    "get_max_prompt_char_length",
    "clean_tags",
    "clean_text",
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

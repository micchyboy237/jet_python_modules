# import nltk
import re
import os
import string
from nltk.tokenize import sent_tokenize
from unidecode import unidecode

# nltk.download("punkt")


def remove_non_alphanumeric(text):
    return re.sub(r'[^a-zA-Z0-9\s-](?<!-)|(?<![\w])-(?![\w])', '', text)


def fix_and_unidecode(text: str) -> str:
    """Converts escaped sequences to actual characters before unidecoding,
    while preserving direct Unicode characters."""

    # Decode only escaped sequences (e.g., "\u00e9" → "é"), keep direct Unicode intact
    def decode_match(match):
        return bytes(match.group(0), "utf-8").decode("unicode_escape")

    fixed_text = re.sub(
        r'\\u[0-9A-Fa-f]{4}|\\x[0-9A-Fa-f]{2}', decode_match, text
    )

    return unidecode(fixed_text)


def has_non_ascii(text: str) -> bool:
    return any(ord(char) >= 128 for char in text)


def find_word_indexes(text: str, word: str) -> list[tuple[int, int]]:
    """
    Returns a list of [start, end] indexes where the word appears in the text.

    :param text: The input text to search within.
    :param word: The word to find in the text.
    :return: A list of [start, end] indexes.
    """
    indexes = []
    word_length = len(word)

    for i in range(len(text) - word_length + 1):
        if text[i:i + word_length] == word:
            indexes.append([i, i + word_length - 1])

    return indexes


def find_sentence_indexes(text: str, word: str) -> list[tuple[int, int]]:
    """
    Returns a list of [start, end] character indexes for sentences that contain the specified word.

    :param text: The input text containing multiple sentences.
    :param word: The word to search for.
    :return: A list of [start, end] character indexes of matching sentences.
    """
    sentences = sent_tokenize(text)  # Split text into sentences
    indexes = []
    start = 0

    for sentence in sentences:
        end = start + len(sentence) - 1  # Compute end index
        if word in sentence:
            indexes.append([start, end])
        start = end + 2  # Move to the next sentence, +2 accounts for period + space

    return indexes


def extract_word_sentences(text: str, word: str) -> list[str]:
    """
    Returns a list of sentences that contain the specified word.

    :param text: The input text containing multiple sentences.
    :param word: The word to search for.
    :return: A list of sentences that contain the word.
    """
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    return [sentence for sentence in sentences if word in sentence]


def extract_substrings(text: str, indexes: list[tuple[int, int]]) -> list[str]:
    """
    Extracts substrings from a text using a list of [start, end] indexes.

    :param text: The input text.
    :param indexes: A list of [start, end] positions.
    :return: A list of extracted substrings.
    """
    return [text[start:end + 1] for start, end in indexes]


def remove_substring(text: str, start: int, end: int) -> str:
    """
    Remove a substring from text given start and end indices (end is exclusive).

    Args:
        text: Input string to modify
        start: Starting index (inclusive)
        end: Ending index (exclusive)

    Returns:
        Modified string with substring removed, or original string if indices are invalid
    """
    if not text or start < 0 or end > len(text) or start > end:
        return text
    return text[:start] + text[end:]


def format_file_path(text: str) -> str:
    # Replace newlines with underscores first
    result = text.replace('\n', '_')
    # Replace non-alphanumeric characters (except spaces) with underscore and convert to lowercase
    result = re.sub(r'[^\w\s]', '_', result.lower())
    # Replace one or more spaces with a single underscore
    result = re.sub(r'\s+', '_', result)
    # Replace multiple consecutive underscores with a single underscore
    result = re.sub(r'_+', '_', result)
    # Remove leading and trailing underscores
    return result.strip('_')

def format_sub_dir(text: str) -> str:
    return format_file_path(text)


def format_sub_source_dir(source: str) -> str:
    """Format a source (URL or file path) into a directory name."""
    clean_source = re.sub(r'^(https?://|www\.)|(\?.*)', '', source)
    clean_source = clean_source.replace(os.sep, '_')
    trans_table = str.maketrans({p: '_' for p in string.punctuation})
    formatted = clean_source.translate(trans_table).lower()
    formatted = re.sub(r'_+', '_', formatted)
    return formatted.strip('_')


__all__ = [
    "fix_and_unidecode",
    "has_non_ascii",
    "find_word_indexes",
    "find_sentence_indexes",
    "extract_word_sentences",
    "extract_substrings",
    "remove_substring",
    "format_sub_dir",
    "format_sub_source_dir",
]

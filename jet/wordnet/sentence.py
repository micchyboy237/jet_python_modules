from io import StringIO
import re
from typing import Callable, List, Optional, Tuple

from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from jet.wordnet.words import count_words, get_words

# nltk.download('punkt')


def process_sentence_newlines(sentences):
    processed_sentences = []
    carry_over_text = ''

    for sentence in sentences:
        newline_index = sentence.find('NEWLINE_TOKEN')

        if newline_index != -1:
            # Add the sentence up to the NEWLINE_TOKEN with any carry-over text
            processed_sentences.append(
                (carry_over_text + ' ' + sentence[:newline_index]).strip())

            # Update carry_over_text with the text after NEWLINE_TOKEN
            carry_over_text = sentence[newline_index:].strip()
        else:
            # Add the sentence with any carry-over text
            processed_sentences.append(
                (carry_over_text + ' ' + sentence).strip())
            carry_over_text = ''  # Reset carry-over text

    # Add the last carry-over text if it's not empty
    if carry_over_text and carry_over_text.replace('NEWLINE_TOKEN', '').strip():
        processed_sentences.append(carry_over_text)

    return processed_sentences


def handle_long_sentence(sentence, count_tokens_func, max_tokens):
    words = sentence.split()
    n = len(words)

    # Generate all possible incremental substrings
    substrings = [' '.join(words[:i + 1]) for i in range(n)]

    # Count tokens for all substrings at once
    token_counts = [count_tokens_func(substring) for substring in substrings]

    segments = []
    start = 0

    while start < n:
        # Find the longest valid substring within max_tokens
        end = start
        while end < n and token_counts[end] - (token_counts[start - 1] if start > 0 else 0) <= max_tokens:
            end += 1

        if end == start:  # If a single word exceeds max_tokens
            segments.append(words[start])
            start += 1
        else:
            segments.append(' '.join(words[start:end]))
            start = end  # Move to the next segment

    return ' '.join(segments)


def get_list_marker_pattern_substring():
    roman_numeral = r'(X{0,3}(IX|IV|V?I{0,3}))'
    ordered_pattern = r'(\d+|[a-zA-Z]|' + roman_numeral + r')[\.\)]+\s?'
    unordered_pattern = r'[-*+]\s?'
    return ordered_pattern + '|' + unordered_pattern


def get_ordered_list_marker_pattern_substring():
    roman_numeral = r'(X{0,3}(IX|IV|V?I{0,3}))'
    ordered_pattern = r'(\d+|[a-zA-Z]|' + roman_numeral + r')[\.\)]+\s?'
    return ordered_pattern


def get_list_marker_pattern():
    pattern = get_list_marker_pattern_substring() + r'$'
    return pattern


def get_list_sentence_pattern():
    pattern = get_list_marker_pattern_substring() + r'[\w]+'
    return pattern


def is_ordered_list_marker(marker):
    pattern = get_ordered_list_marker_pattern_substring() + r'$'
    return bool(re.match(pattern, marker.strip(), re.IGNORECASE))


def is_ordered_list_sentence(sentence):
    pattern = get_ordered_list_marker_pattern_substring() + r'[\w]+'
    return bool(re.match(pattern, sentence.strip(), re.IGNORECASE))


def is_list_marker(marker):
    ordered_pattern = r'^(\d+|[a-zA-Z]|' + \
        r'(X{0,3}(IX|IV|V?I{0,3}))' + r')[\.\)]+\s?$'
    unordered_pattern = r'^[-*+]\s?$'
    pattern = r'(' + ordered_pattern + r'|' + unordered_pattern + r')'
    return bool(re.match(pattern, marker.strip(), re.IGNORECASE))


def is_list_sentence(sentence):
    pattern = get_list_sentence_pattern()
    return not is_list_marker(sentence) and bool(re.match(pattern, sentence.strip(), re.IGNORECASE))


def is_unordered_list_marker(marker):
    pattern = r'^[-*+]\s?$'
    return bool(re.match(pattern, marker.strip(), re.IGNORECASE))


def is_last_word_in_sentence(word, text):
    word = word.strip(".,!?\"'”)").lower()
    sentences = split_sentences(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        real_words = [w.strip(".,!?\"'”)").lower()
                      for w in words if any(c.isalnum() for c in w)]
        if real_words and real_words[-1] == word:
            return True
    return False


def is_sentence(text: str) -> bool:
    """Validate if the input text is a single, well-formed sentence using NLTK, with no newlines."""
    # Strip whitespace for consistent comparison
    text = text.strip()
    if not text:
        return False

    # Check for newlines (\n, \r, or \r\n)
    if any(newline in text for newline in ['\n', '\r']):
        return False

    # Use NLTK's sent_tokenize to check if text is recognized as a single sentence
    sentences: list[str] = sent_tokenize(text)
    if len(sentences) != 1:
        return False

    # Verify the tokenized sentence matches the input (ignoring whitespace)
    if sentences[0].strip() != text:
        return False

    # Additional heuristic: Ensure at least one word token exists
    tokens: list[str] = word_tokenize(text)
    if not tokens:
        return False

    # Optional heuristic: Check for ending punctuation (., !, ?)
    # This is lenient to allow flexibility (e.g., for informal text)
    has_punctuation: bool = text[-1] in '.!?'

    # Ensure there is at least one non-punctuation token
    non_punct_tokens: list[str] = [t for t in tokens if t not in '.!?,;:']
    return len(non_punct_tokens) > 0 and has_punctuation


def split_sentences(text: str, num_sentence: int = 1) -> list[str]:
    if num_sentence < 1:
        raise ValueError("num_sentence must be a positive integer")

    # Split text by newlines first to treat each line as a potential sentence
    lines = text.split('\n')
    sentences = []
    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Apply sent_tokenize to each line to handle punctuation-based splitting
            line_sentences = sent_tokenize(line.strip())
            sentences.extend(line_sentences)

    adjusted_sentences = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i]

        if is_list_marker(current_sentence) and i + 1 < len(sentences):
            combined = current_sentence + ' ' + sentences[i + 1]
            if is_list_sentence(combined):
                adjusted_sentences.append(combined)
                i += 2
                continue
        elif is_list_sentence(current_sentence):
            adjusted_sentences.append(current_sentence)
        else:
            adjusted_sentences.append(current_sentence)

        i += 1

    # Combine sentences based on num_sentence
    combined_results = []
    for j in range(0, len(adjusted_sentences), num_sentence):
        chunk = adjusted_sentences[j:j + num_sentence]
        combined_results.append('\n'.join(chunk))

    return combined_results


def split_sentences_with_separators(text: str, num_sentence: int = 1) -> List[str]:
    """Split text into sentences, preserving the separator after each sentence.

    Args:
        text: Input text to split.
        num_sentence: Number of sentences to combine into each chunk.

    Returns:
        List of strings, each containing combined sentences with their trailing separator.
    """
    if num_sentence < 1:
        raise ValueError("num_sentence must be a positive integer")

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    adjusted_sentences = []
    i = 0
    current_pos = 0
    while i < len(sentences):
        current_sentence = sentences[i]
        # Find the sentence in the original text to determine its separator
        start_idx = text.find(current_sentence, current_pos)
        if start_idx == -1:
            # Fallback if sentence not found
            adjusted_sentences.append((current_sentence, " "))
            i += 1
            continue
        end_idx = start_idx + len(current_sentence)
        # Extract separator
        separator = ""
        if end_idx < len(text):
            next_sentence_idx = text.find(
                sentences[i + 1], end_idx) if i + 1 < len(sentences) else len(text)
            separator = text[end_idx:next_sentence_idx]
            if not separator.strip():  # If separator is only whitespace/newlines
                separator = separator if "\n" in separator else " "
            else:
                separator = " "  # Default to space for non-whitespace separators

        if is_list_marker(current_sentence) and i + 1 < len(sentences):
            combined = current_sentence + ' ' + sentences[i + 1]
            if is_list_sentence(combined):
                # Find the combined sentence in the original text
                combined_start = text.find(combined, current_pos)
                if combined_start != -1:
                    combined_end = combined_start + len(combined)
                    combined_separator = ""
                    if combined_end < len(text):
                        next_idx = text.find(
                            sentences[i + 2], combined_end) if i + 2 < len(sentences) else len(text)
                        combined_separator = text[combined_end:next_idx]
                        if not combined_separator.strip():
                            combined_separator = combined_separator if "\n" in combined_separator else " "
                        else:
                            combined_separator = " "
                    adjusted_sentences.append((combined, combined_separator))
                    current_pos = combined_end
                    i += 2
                    continue
        elif is_list_sentence(current_sentence):
            adjusted_sentences.append((current_sentence, separator))
        else:
            adjusted_sentences.append((current_sentence, separator))

        current_pos = end_idx
        i += 1

    # Combine sentences based on num_sentence
    combined_results = []
    for j in range(0, len(adjusted_sentences), num_sentence):
        chunk = adjusted_sentences[j:j + num_sentence]
        combined_sentence = ""
        for k, (sentence, sep) in enumerate(chunk):
            combined_sentence += sentence
            if k < len(chunk) - 1:  # Add separator except for the last sentence
                combined_sentence += sep
        # Use the last sentence's separator for the combined chunk
        final_separator = chunk[-1][1] if chunk else " "
        combined_results.append(combined_sentence + final_separator)

    return combined_results


def adaptive_split(text, count_tokens_func=count_words, max_tokens=0):
    # Check for empty or whitespace-only strings
    if not text.strip():
        return []

    max_tokens = max_tokens or 0

    # Replace \n with NEWLINE_TOKEN
    text = text.replace('\n', 'NEWLINE_TOKEN')

    # Use NLTK for more robust sentence splitting
    raw_sentences = sent_tokenize(text)
    processed_sentences = process_sentence_newlines(raw_sentences)

    segments = []
    current_segment = ''

    for sentence in processed_sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue

        # Convert NEWLINE_TOKEN to \n before counting tokens
        sentence_for_token_count = sentence.replace('NEWLINE_TOKEN', '\n')
        current_segment_for_token_count = current_segment.replace(
            'NEWLINE_TOKEN', '\n')

        # Calculate token counts
        current_tokens_len = count_tokens_func(current_segment_for_token_count)
        tokens_len = count_tokens_func(sentence_for_token_count)

        stripped_sentence = sentence.replace('NEWLINE_TOKEN', '')
        # Check if the sentence is a continuation of an ordered list item
        if is_ordered_list_marker(stripped_sentence):
            if is_ordered_list_sentence(stripped_sentence):
                # Append the current segment if it's not empty
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ''

                segments.append(sentence)
            else:
                current_segment += (' ' if current_segment else '') + sentence
        # Check if adding the sentence exceeds the max token limit
        elif current_tokens_len + tokens_len <= max_tokens:
            current_segment += (' ' if current_segment else '') + sentence
        else:
            # Append the current segment if it's not empty
            if current_segment and not is_ordered_list_marker(current_segment):
                segments.append(current_segment)
                current_segment = ''

            # Handle the next sentence
            if tokens_len <= max_tokens:
                current_segment = sentence
            else:
                # Split long sentence into words and build segments
                long_segment = handle_long_sentence(
                    sentence, count_tokens_func, max_tokens)

                current_segment += (' ' if current_segment else '') + \
                    long_segment

    # Append the last segment if it exists
    if current_segment:
        segments.append(current_segment)

    # Replace NEWLINE_TOKEN with \n on all segments
    segments = [segment.replace('NEWLINE_TOKEN', '\n') for segment in segments]

    return segments


def get_unique_sentences(text: str) -> list[str]:
    """
    Get unique sentences from text using split_sentences.

    Args:
        text (str): Input text to process

    Returns:
        list[str]: List of unique sentences
    """
    sentences = split_sentences(text)
    return list(dict.fromkeys(sentences))


def merge_sentences(sentences: list[str], max_tokens: int) -> list[str]:
    merged_sentences: list[str] = []
    current_group: list[str] = []
    current_count: int = 0

    for sentence in sentences:
        sentence_count: int = count_words(sentence)

        # If adding the next sentence exceeds the limit, finalize the current group
        if current_group and current_count + sentence_count > max_tokens:
            merged_sentences.append('\n'.join(current_group))
            current_group = []
            current_count = 0

        current_group.append(sentence)
        current_count += sentence_count

    # Add any remaining sentences
    if current_group:
        merged_sentences.append('\n'.join(current_group))

    return merged_sentences


def group_sentences(text: str, max_tokens: int):
    splitted_sentences = split_sentences(text)
    grouped_sentences = merge_sentences(splitted_sentences, max_tokens)
    return grouped_sentences


def count_sentences(text):
    sentences = split_sentences(text)

    return len(sentences)


def get_sentences(text: str, n: int) -> list[str]:
    sentences = split_sentences(text, n)

    return sentences


def split_by_punctuations(text: str, punctuations: list[str]) -> list[str]:
    if not text:
        return []
    if not punctuations:
        raise ValueError("Punctuation list cannot be empty or None.")

    pattern = f"[{''.join(map(re.escape, punctuations))}]"
    return [segment.strip() for segment in re.split(pattern, text) if segment.strip()]


def encode_text_to_strings(text: str, tokenize: Optional[Callable] = None) -> list[str]:
    """
    Tokenizes text and returns a list of token strings.

    Uses existing sentence splitting and token counting for consistency.
    """
    if not text.strip():
        return []

    if not tokenize:
        tokens = get_words(text)
    else:
        tokens = tokenize(text)

    return tokens


__all__ = [
    "process_sentence_newlines",
    "handle_long_sentence",
    "get_list_marker_pattern_substring",
    "get_list_marker_pattern",
    "get_list_sentence_pattern",
    "is_ordered_list_marker",
    "is_ordered_list_sentence",
    "adaptive_split",
    "split_sentences",
    "split_sentences_with_separators",
    "merge_sentences",
    "group_sentences",
    "count_sentences",
    "get_sentences",
    "split_by_punctuations",
    "encode_text_to_strings",
    "is_last_word_in_sentence",
]

if __name__ == "__main__":
    text = 'Ang mga pang-uri o adjectives sa Ingles ay salitang nagbibigay turing o naglalarawan sa isang pangngalan o panghalip. Ito ay nagsasaad ng uri o katangian ng tao, bagay, hayop, pook, o pangyayari.'

    sentence_count = count_sentences(text)
    print(f"Number of sentences: {sentence_count}")

    sentences = get_sentences(text, 3)
    print(sentences)

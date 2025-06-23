import re

from typing import List, Optional, TypedDict
from transformers import PreTrainedTokenizer

from jet.utils.text import remove_substring


class NgramRepeat(TypedDict):
    ngram: str
    start_index: int
    end_index: int
    full_end_index: int
    num_of_repeats: int


def find_repeated_consecutive_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
    case_sensitive: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> List[NgramRepeat]:
    words_with_pos = []
    if tokenizer:
        tokens = tokenizer.tokenize(text)
        words_with_pos = []
        current_pos = 0
        for token in tokens:
            token_text = tokenizer.convert_tokens_to_string([token])
            match = re.search(re.escape(token_text), text[current_pos:])
            if match:
                start = current_pos + match.start()
                end = current_pos + match.end()
                words_with_pos.append((token_text, start, end))
                current_pos = end
    else:
        pattern = r"\b(?:\w+['’]\w+|\w+[.,!?;]?)\b"
        matches = list(re.finditer(pattern, text))
        for match in matches:
            word = match.group(0)
            start = match.start()
            end = match.end()
            words_with_pos.append((word, start, end))

    def clean_word(w: str) -> str:
        return w if case_sensitive else w.lower()

    words_to_compare = [clean_word(w[0]) for w in words_with_pos]
    max_words = max_words or len(words_to_compare)
    results = []
    for n in range(min_words, max_words + 1):
        i = 0
        while i <= len(words_to_compare) - n * min_repeat:
            count = 1
            while (
                i + (count + 1) * n <= len(words_to_compare)
                and words_to_compare[i: i + n] == words_to_compare[i + count * n: i + (count + 1) * n]
            ):
                count += 1
            if count >= min_repeat:
                start_char = words_with_pos[i][1]
                end_char = words_with_pos[i + n - 1][2]
                full_end_char = words_with_pos[i + count * n - 1][2]
                ngram_text = text[start_char:end_char] if tokenizer else " ".join(
                    w[0] for w in words_with_pos[i: i + n])
                results.append(
                    NgramRepeat(
                        ngram=ngram_text,
                        start_index=start_char,
                        end_index=end_char,
                        full_end_index=full_end_char,
                        num_of_repeats=count,
                    )
                )
                i += count * n
            else:
                i += 1
    return results


def clean_repeated_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
    case_sensitive: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> str:
    """
    Clean repeated consecutive n-grams from the text.

    Args:
        text: Input text to process.
        min_words: Minimum number of words in an n-gram.
        max_words: Maximum number of words in an n-gram (None for no limit).
        min_repeat: Minimum number of consecutive repeats to consider.
        case_sensitive: Whether to treat case differences as distinct.
        tokenizer: Optional tokenizer for custom word splitting.

    Returns:
        Text with repeated n-grams removed, keeping the first occurrence.
    """
    repeats = find_repeated_consecutive_ngrams(
        text,
        min_words=min_words,
        max_words=max_words,
        min_repeat=min_repeat,
        case_sensitive=case_sensitive,
        tokenizer=tokenizer,
    )
    if not repeats:
        return text

    # Remove all repeats from end_index + 1 to full_end_index safely on text
    # We'll process from the end to avoid messing up indices as we remove substrings.
    result = text
    # Convert repeats to a list of (start, end) tuples to remove
    # We want to remove from end_index (exclusive) to full_end_index (exclusive)
    # i.e., result[end_index:full_end_index] should be removed
    # But since end_index is inclusive in the repeat, we want to start at end_index, but not include the ngram itself
    # So, remove from end_index to full_end_index

    # To avoid index shifting, process from the end
    spans_to_remove = []
    for repeat in repeats:
        start = repeat["end_index"]
        end = repeat["full_end_index"]
        if start < end:
            spans_to_remove.append((start, end))

    for start, end in spans_to_remove:
        result = remove_substring(result, start, end)

    return result.strip()  # Strip to handle any trailing spaces

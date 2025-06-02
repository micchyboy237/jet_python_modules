from typing import List, Optional, TypedDict


class NgramRepeat(TypedDict):
    ngram: str
    start_index: int  # char index of first character
    end_index: int    # char index of last character (inclusive)
    full_end_index: int  # last char index of entire repeated block
    num_of_repeats: int


def find_repeated_consecutive_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
) -> List[NgramRepeat]:
    """
    Detects repeated consecutive n-grams in a single string.
    Returns a list of dicts with keys: ngram, start_index, end_index (char offsets), num_of_repeats.

    Args:
        text: The input text string.
        min_words: Minimum size of n-gram (number of words).
        max_words: Maximum size of n-gram (number of words).
        min_repeat: Minimum number of consecutive repeats to detect.

    Returns:
        List of NgramRepeat dicts with keys:
        - 'ngram': repeated n-gram string
        - 'start_index': starting char index in text
        - 'end_index': ending char index in text (inclusive)
        - 'num_of_repeats': how many times the n-gram repeats consecutively
    """
    # Split text into words with their character positions
    words = []
    pos = 0
    text_len = len(text)
    while pos < text_len:
        # Skip whitespace
        while pos < text_len and text[pos].isspace():
            pos += 1
        if pos >= text_len:
            break
        start = pos
        while pos < text_len and not text[pos].isspace():
            pos += 1
        end = pos  # exclusive
        words.append((text[start:end], start, end))

    results: List[NgramRepeat] = []
    max_words = max_words or len(words)
    results = []

    for i in range(len(words)):
        for n in range(min_words, max_words + 1):
            if i + n * min_repeat > len(words):
                continue

            count = 1
            while (
                i + (count + 1) * n <= len(words)
                and all(
                    words[i + offset][0] == words[i + count * n + offset][0]
                    for offset in range(n)
                )
            ):
                count += 1

            if count >= min_repeat:
                start_char = words[i][1]
                end_char = words[i + n - 1][2] - 1
                full_end_char = words[i + count * n - 1][2] - 1

                results.append(
                    NgramRepeat(
                        ngram=" ".join(w[0] for w in words[i: i + n]),
                        start_index=start_char,
                        end_index=end_char,
                        full_end_index=full_end_char,
                        num_of_repeats=count,
                    )
                )

    # Optional: sort results by start_index, then ngram length ascending
    results.sort(key=lambda r: (r["start_index"], len(r["ngram"].split())))
    return results

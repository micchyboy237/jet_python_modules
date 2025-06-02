from typing import List, Optional, TypedDict


class NgramRepeat(TypedDict):
    ngram: str
    start_index: int
    end_index: int
    num_of_repeats: int


def find_repeated_consecutive_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
) -> List[NgramRepeat]:
    """
    Detects repeated consecutive n-grams in a single string.
    Returns a list of dicts with keys: ngram, start_index, end_index, num_of_repeats.

    Args:
        text: The input text string.
        min_words: Minimum size of n-gram (number of words).
        max_words: Maximum size of n-gram (number of words).
        min_repeat: Minimum number of consecutive repeats to detect.

    Returns:
        List of NgramRepeat dicts with keys:
        - 'ngram': repeated n-gram string
        - 'start_index': starting word index
        - 'end_index': ending word index (inclusive)
        - 'num_of_repeats': how many times the n-gram repeats consecutively
    """
    words = text.split()
    results: List[NgramRepeat] = []
    max_words = max_words or len(words)

    i = 0
    while i < len(words):
        longest_match: NgramRepeat | None = None

        for n in range(max_words, min_words - 1, -1):
            if i + n * min_repeat > len(words):
                continue

            count = 1
            while (
                i + (count + 1) * n <= len(words)
                and words[i:i + n] == words[i + count * n:i + (count + 1) * n]
            ):
                count += 1

            if count >= min_repeat:
                candidate = NgramRepeat(
                    ngram=" ".join(words[i:i + n]),
                    start_index=i,
                    end_index=i + count * n - 1,
                    num_of_repeats=count,
                )
                if longest_match is None or (candidate["end_index"] - candidate["start_index"] >
                                             longest_match["end_index"] - longest_match["start_index"]):
                    longest_match = candidate

        if longest_match:
            results.append(longest_match)
            i = longest_match["end_index"] + 1
        else:
            i += 1

    return results

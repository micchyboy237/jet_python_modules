import string
from typing import List, Optional, TypedDict


class NgramRepeat(TypedDict):
    ngram: str
    start_index: int  # char index of first character of first ngram occurrence
    end_index: int    # char index of last character of first ngram occurrence
    full_end_index: int  # char index of last character of last repeated ngram occurrence
    num_of_repeats: int


def find_repeated_consecutive_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
    case_sensitive: bool = False,
) -> List[NgramRepeat]:
    words_with_pos = []
    index = 0
    for word in text.split():
        start = text.index(word, index)
        end = start + len(word)
        words_with_pos.append((word, start, end))
        index = end

    def clean_word(w: str) -> str:
        return w if case_sensitive else w.lower().strip(string.punctuation)

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
                end_char = words_with_pos[i + n - 1][2] - 1
                full_end_char = words_with_pos[i + count * n - 1][2] - 1

                results.append(
                    NgramRepeat(
                        ngram=" ".join(w[0] for w in words_with_pos[i: i + n]),
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

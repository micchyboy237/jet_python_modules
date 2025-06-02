import re
from typing import List, Optional, TypedDict
from transformers import PreTrainedTokenizer


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
        # Use tokenizer for word separation
        tokens = tokenizer.tokenize(text)
        words_with_pos = []
        current_pos = 0
        for token in tokens:
            # Find the token in the original text to get accurate positions
            token_text = tokenizer.convert_tokens_to_string([token])
            match = re.search(re.escape(token_text), text[current_pos:])
            if match:
                start = current_pos + match.start()
                end = current_pos + match.end()
                words_with_pos.append((token_text, start, end))
                current_pos = end
    else:
        # Original word-based splitting logic
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
                # Use original text substring to avoid adding spaces
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

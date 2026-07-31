import re
from typing import List, Optional, TypedDict, Tuple
from transformers import PreTrainedTokenizer
from jet.utils.text import remove_substring
from tqdm import tqdm
from collections import defaultdict


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
    # Early termination for short texts
    if len(text.strip()) < min_words * min_repeat * 2:
        return []

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
    # Cap max_words at 10 unless specified
    max_words = max_words or min(len(words_to_compare), 10)
    if max_words < min_words:
        return []

    results = []
    for n in tqdm(range(min_words, max_words + 1), desc="Processing n-grams"):
        ngram_positions = defaultdict(list)
        # Use sliding window to collect n-grams
        for i in range(len(words_to_compare) - n + 1):
            ngram = tuple(words_to_compare[i:i + n])
            ngram_positions[ngram].append(i)

        for ngram, indices in ngram_positions.items():
            if len(indices) < min_repeat:
                continue
            i = 0
            while i < len(indices):
                count = 1
                start_idx = indices[i]
                j = i + 1
                # Check for consecutive indices
                while j < len(indices) and indices[j] == indices[j - 1] + n:
                    count += 1
                    j += 1
                if count >= min_repeat:
                    start_char = words_with_pos[start_idx][1]
                    end_char = words_with_pos[start_idx + n - 1][2]
                    try:
                        full_end_char = words_with_pos[start_idx +
                                                       count * n - 1][2]
                    except IndexError:
                        # Handle edge case where indices exceed list length
                        break
                    ngram_text = text[start_char:end_char] if tokenizer else " ".join(
                        w[0] for w in words_with_pos[start_idx:start_idx + n]
                    )
                    results.append(
                        NgramRepeat(
                            ngram=ngram_text,
                            start_index=start_char,
                            end_index=end_char,
                            full_end_index=full_end_char,
                            num_of_repeats=count,
                        )
                    )
                i = max(i + 1, j)  # Move to next unprocessed index

    return sorted(results, key=lambda x: (x["start_index"], -x["num_of_repeats"], x["end_index"] - x["start_index"]))


def clean_repeated_ngrams(
    text: str,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_repeat: int = 2,
    case_sensitive: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> str:
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
    result = text
    spans_to_remove = []
    for repeat in repeats:
        start = repeat["end_index"]
        end = repeat["full_end_index"]
        if start < end:
            spans_to_remove.append((start, end))
    # Process from end to start
    for start, end in sorted(spans_to_remove, reverse=True):
        result = remove_substring(result, start, end)
    return result.strip()

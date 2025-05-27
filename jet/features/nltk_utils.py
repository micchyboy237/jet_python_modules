import nltk
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from typing import Union, List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from functools import partial

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('R'):
        return 'r'
    return 'n'


def process_single_text(single_text: str, n: Optional[int] = None, min_count: int = 1,
                        in_sequence: bool = False, max_n: int = 10) -> Dict[str, int]:
    if not single_text.strip():
        return {}
    counts = Counter()
    sentences = sent_tokenize(single_text.lower())
    sentence_words = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [token for token in tokens if token.isalpha()
                 and token not in STOP_WORDS]
        tagged_words = pos_tag(words)
        lemmatized_words = []
        for word, tag in tagged_words:
            pos = get_wordnet_pos(tag)
            # Force verb lemmatization for third-person singular forms
            if pos == 'v' and tag in ('VBZ', 'VBP'):
                lemmatized = LEMMATIZER.lemmatize(word, pos)
                # Additional check for common verb endings
                if word.endswith('s') and lemmatized == word:
                    lemmatized = LEMMATIZER.lemmatize(word[:-1], pos)
                elif word.endswith('es') and lemmatized == word:
                    lemmatized = LEMMATIZER.lemmatize(word[:-2], pos)
            else:
                lemmatized = LEMMATIZER.lemmatize(word, pos)
            lemmatized_words.append(lemmatized)
        if lemmatized_words:
            sentence_words.append(lemmatized_words)
    if not sentence_words:
        return {}
    if n is None:
        max_n_local = min(max(len(words) for words in sentence_words), max_n)
        combination_sizes = range(1, max_n_local + 1)
    else:
        combination_sizes = [min(n, max_n)]
    for lemmatized_words in sentence_words:
        for current_n in combination_sizes:
            if current_n > len(lemmatized_words):
                continue
            if in_sequence:
                combinations = [
                    ','.join(lemmatized_words[i:i + current_n])
                    for i in range(len(lemmatized_words) - current_n + 1)
                ]
            else:
                combinations = [','.join(c) for c in itertools.combinations(
                    lemmatized_words, current_n)]
            counts.update(combinations)
    result = dict(sorted(
        {ngram: count for ngram, count in counts.items() if count >=
         min_count}.items(),
        key=lambda x: x[1], reverse=True
    ))
    return result


def process_wrapper(text_item: str, n: Optional[int] = None, min_count: int = 1,
                    in_sequence: bool = False, max_n: int = 10) -> Dict[str, int]:
    return process_single_text(text_item, n, min_count, in_sequence, max_n)


def get_word_sentence_combination_counts(
    text: Union[str, List[str]],
    n: Optional[int] = None,
    min_count: int = 1,
    in_sequence: bool = False,
    max_n: int = 10,
    show_progress: bool = True
) -> Union[Dict[str, int], List[Dict[str, int]]]:
    if isinstance(text, str):
        return process_single_text(text, n, min_count, in_sequence, max_n)
    elif isinstance(text, list):
        if len(text) <= 10 or n == 1:
            results = [
                process_wrapper(t, n, min_count, in_sequence, max_n) for t in text
            ]
            return results  # Include empty dictionaries
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                process_func = partial(
                    process_wrapper,
                    n=n,
                    min_count=min_count,
                    in_sequence=in_sequence,
                    max_n=max_n
                )
                results = list(tqdm(
                    executor.map(process_func, text),
                    total=len(text),
                    desc="Processing texts",
                    disable=not show_progress
                ))
                return results  # Include empty dictionaries
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                raise
    else:
        raise ValueError("Input must be a string or a list of strings")


def process_single_text_simple(single_text: str) -> Dict[str, int]:
    if not single_text.strip():
        return {}
    tokens = word_tokenize(single_text.lower())
    words = [token for token in tokens if token.isalpha()
             and token not in STOP_WORDS]
    if not words:
        return {}
    tagged_words = pos_tag(words)
    lemmatized_words = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words
    ]
    counts = Counter(lemmatized_words)
    result = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return result


def get_word_counts_lemmatized(text: Union[str, List[str]]) -> Union[Dict[str, int], List[Dict[str, int]]]:
    if isinstance(text, str):
        result = process_single_text_simple(text)
        return result if result else {}
    elif isinstance(text, list):
        results = [process_single_text_simple(t) for t in text]
        return [r for r in results if r]
    else:
        raise ValueError("Input must be a string or a list of strings")

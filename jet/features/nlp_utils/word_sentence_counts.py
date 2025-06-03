import nltk
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from typing import Union, List, Dict, Tuple, Optional
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
                        in_sequence: bool = False, max_n: int = 10) -> Dict[Tuple[str, ...], int]:
    counts = Counter()
    sentences = sent_tokenize(single_text.lower())
    sentence_words = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [token for token in tokens if token.isalpha()
                 and token not in STOP_WORDS]
        tagged_words = pos_tag(words)
        lemmatized_words = [
            LEMMATIZER.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words
        ]
        if lemmatized_words:
            sentence_words.append(lemmatized_words)
    if n is None:
        max_n_local = min(
            max(len(words) for words in sentence_words) if sentence_words else 1, max_n)
        combination_sizes = range(1, max_n_local + 1)
    else:
        combination_sizes = [min(n, max_n)]
    for lemmatized_words in sentence_words:
        for current_n in combination_sizes:
            if current_n > len(lemmatized_words):
                continue
            if in_sequence:
                combinations = [
                    tuple(lemmatized_words[i:i + current_n])
                    for i in range(len(lemmatized_words) - current_n + 1)
                ]
            else:
                combinations = list(itertools.combinations(
                    lemmatized_words, current_n))
            counts.update(combinations)
    return dict(sorted(
        {ngram: count for ngram, count in counts.items() if count >=
         min_count}.items(),
        key=lambda x: x[1], reverse=True
    ))


def process_wrapper(text_item: str, n: Optional[int] = None, min_count: int = 1,
                    in_sequence: bool = False, max_n: int = 10) -> Dict[Tuple[str, ...], int]:
    return process_single_text(text_item, n, min_count, in_sequence, max_n)


def get_word_sentence_combination_counts(
    text: Union[str, List[str]],
    n: Optional[int] = None,
    min_count: int = 1,
    in_sequence: bool = False,
    max_n: int = 10,
    show_progress: bool = True
) -> Union[Dict[Tuple[str, ...], int], List[Dict[Tuple[str, ...], int]]]:
    if isinstance(text, str):
        return process_single_text(text, n, min_count, in_sequence, max_n)
    elif isinstance(text, list):
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                process_func = partial(
                    process_wrapper, n=n, min_count=min_count, in_sequence=in_sequence, max_n=max_n)
                results = list(tqdm(
                    executor.map(process_func, text),
                    total=len(text),
                    desc="Processing texts",
                    disable=not show_progress
                ))
                return results
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                raise
    else:
        raise ValueError("Input must be a string or a list of strings")

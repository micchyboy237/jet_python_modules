import itertools
import multiprocessing
from typing import Optional, Union, Dict, List
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from .utils import get_wordnet_pos

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def process_single_text(single_text: str, n: Optional[int] = None, min_count: int = 1,
                        in_sequence: bool = False, max_n: int = 10) -> Dict[str, int]:
    if not single_text.strip():
        return {}

    counts = Counter()
    sentences = sent_tokenize(single_text.lower())
    sentence_words = []

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [t for t in tokens if (
            t.isalpha() or t.isdigit()) and t not in STOP_WORDS]
        tagged = pos_tag(words)
        lemmatized = []

        for word, tag in tagged:
            pos = get_wordnet_pos(tag)
            lemma = LEMMATIZER.lemmatize(word, pos)
            if pos == 'v' and tag in ('VBZ', 'VBP') and lemma == word:
                if word.endswith('es'):
                    lemma = LEMMATIZER.lemmatize(word[:-2], pos)
                elif word.endswith('s'):
                    lemma = LEMMATIZER.lemmatize(word[:-1], pos)
            lemmatized.append(lemma)

        if lemmatized:
            sentence_words.append(lemmatized)

    if not sentence_words:
        return {}

    if n is None:
        max_n_local = min(max(len(w) for w in sentence_words), max_n)
        sizes = range(1, max_n_local + 1)
    else:
        sizes = [min(n, max_n)]

    for words in sentence_words:
        for size in sizes:
            if size > len(words):
                continue
            if in_sequence:
                combs = [','.join(words[i:i + size])
                         for i in range(len(words) - size + 1)]
            else:
                combs = [','.join(c)
                         for c in itertools.combinations(words, size)]
            counts.update(combs)

    return dict(sorted({k: v for k, v in counts.items() if v >= min_count}.items(), key=lambda x: x[1], reverse=True))


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

    if isinstance(text, list):
        max_workers = 2
        process_func = partial(process_single_text, n=n,
                               min_count=1, in_sequence=in_sequence, max_n=max_n)
        if len(text) <= 10 or n == 1:
            all_counts = [process_func(t) for t in text]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                all_counts = list(tqdm(executor.map(process_func, text), total=len(
                    text), disable=not show_progress))

        total = Counter()
        for d in all_counts:
            total.update(d)
        valid = {k for k, v in total.items() if v >= min_count}

        results = []
        for d in all_counts:
            filtered = {k: v for k, v in d.items() if k in valid}
            results.append(
                dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)))
        return results

    raise ValueError("Input must be a string or list of strings")

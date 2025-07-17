import re
from typing import List, Tuple, Dict, Union
from collections import defaultdict
import spacy
import logging
from itertools import combinations
from jet.logger import logger

nlp = spacy.load("en_core_web_sm")


def generate_ngrams(tokens: List[str], ngram_range: Tuple[int, int] = (1, 1)) -> List[str]:
    """Generate n-grams from a list of tokens within the specified range."""
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            ngrams.append(ngram)
    return ngrams


def tokenize_document(doc: str, ngram_range: Tuple[int, int] = (1, 1)) -> List[str]:
    """Tokenize a document into lowercase alphabetic, numeric, or alphanumeric n-grams with specific POS tags."""
    spacy_doc = nlp(doc)
    allowed_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    valid_token_pattern = re.compile(r'^[a-zA-Z0-9]+$')

    # Get single tokens that match POS and pattern
    tokens = [
        token.text.lower() for token in spacy_doc
        if token.pos_ in allowed_pos and valid_token_pattern.match(token.text)
    ]

    # Generate n-grams from valid tokens
    return generate_ngrams(tokens, ngram_range)


def find_cooccurring_words(
    documents: List[str],
    min_docs: int = 2,
    ngram_range: Tuple[int, int] = (1, 1)
) -> List[Tuple[str, str, int]]:
    """Find co-occurring n-grams (words or phrases) across documents."""
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for i, doc in enumerate(documents):
        ngrams = set(tokenize_document(doc, ngram_range))
        for ngram1, ngram2 in combinations(sorted(ngrams), 2):
            if ngram1 in ngram2.split() or ngram2 in ngram1.split():
                continue
            pair_counts[(ngram1, ngram2)] += 1
    result = [
        (ngram1, ngram2, count)
        for (ngram1, ngram2), count in pair_counts.items()
        if count >= min_docs
    ]
    # Sort by count (descending), then prioritize pairs with bigrams, then alphabetically

    def sort_key(pair: Tuple[str, str, int]) -> Tuple[int, int, int, str, str]:
        ngram1, ngram2, count = pair
        # Count number of words in each n-gram (1 for unigram, 2 for bigram, etc.)
        len_ngram1 = len(ngram1.split())
        len_ngram2 = len(ngram2.split())
        # Prioritize pairs with at least one bigram (max length), then alphabetically
        return (-count, -(len_ngram1 + len_ngram2), -max(len_ngram1, len_ngram2), ngram1, ngram2)
    sorted_result = sorted(result, key=sort_key)
    return sorted_result

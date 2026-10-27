import re
from typing import List, Union

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded once
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Pre-compiled regex for tokenization (matches alphanumeric sequences)
# This is significantly faster than split() or regex with boundaries
TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)

# Load stop words into a frozenset for O(1) lookups
STOP_WORDS = frozenset(stopwords.words("english"))


def preprocess_texts(texts: Union[str, List[str]]) -> List[str]:
    """
    A minimal, high-performance preprocessor for BM25.
    - Lowercases text.
    - Extracts alphanumeric tokens using regex.
    - Removes standard English stopwords.
    - Joins tokens with spaces.

    Args:
        texts: A single string or a list of strings.

    Returns:
        A list of preprocessed strings.
    """
    if isinstance(texts, str):
        texts = [texts]

    processed = []
    for text in texts:
        # 1. Lowercase
        lower_text = text.lower()

        # 2. Tokenize using regex (much faster than NLTK/spacy)
        tokens = TOKEN_PATTERN.findall(lower_text)

        # 3. Filter stopwords
        filtered_tokens = [t for t in tokens if t not in STOP_WORDS]

        # 4. Join back into a string for BM25 input
        processed.append(" ".join(filtered_tokens))

    return processed

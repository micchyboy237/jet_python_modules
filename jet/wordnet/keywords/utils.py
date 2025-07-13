import re
from typing import Dict, List, Optional, Sequence, Set, Tuple
from nltk.corpus import stopwords

en_stopwords = stopwords.words("english")


def expand_tokens_with_subtokens(tokens: Set[str]) -> Set[str]:
    """Get subtokens from a list of tokens., filtering for stopwords."""
    results = set()
    for token in tokens:
        results.add(token)
        sub_tokens = re.findall(r"\w+", token)
        if len(sub_tokens) > 1:
            results.update(
                {w for w in sub_tokens if w not in en_stopwords})

    return results


def preprocess_text(text: str) -> str:
    """Preprocess text for keyword extraction by removing all punctuation."""
    # Remove all punctuation except for decimal points in numbers
    # First, preserve decimal points by temporarily replacing them
    text = re.sub(r'(\d)\.(\d)', r'\1DECIMALPOINT\2', text)
    # Remove all punctuation
    cleaned = re.sub(r'[^\w\s]', '', text)
    # Restore decimal points
    cleaned = cleaned.replace('DECIMALPOINT', '.')
    # Remove excessive whitespace and normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    return cleaned

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
    """Preprocess text for keyword extraction while preserving original content."""
    # Remove excessive whitespace and normalize spaces
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Replace multiple punctuation marks with single (e.g., '!!!' -> '!')
    cleaned = re.sub(r'([!?.]){2,}', r'\1', cleaned)
    # Ensure consistent spacing around punctuation, but preserve decimal points
    cleaned = re.sub(r'\s*([,!?;:])\s*', r' \1 ', cleaned)
    cleaned = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2',
                     cleaned)  # Preserve decimal points
    # Ensure space before final punctuation if it exists
    if cleaned and cleaned[-1] in ',!?;.':
        cleaned = cleaned[:-1] + ' ' + cleaned[-1]
    return cleaned

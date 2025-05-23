import re
from collections import Counter
from typing import Optional, Set

from jet.wordnet.keywords.utils import expand_tokens_with_subtokens, en_stopwords


def simple_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """Extract keywords with simple algorithm."""
    tokens = [t.strip().lower() for t in re.findall(r"\w+", text_chunk)]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in en_stopwords]

    token_counts = Counter(tokens)
    keywords = [keyword for keyword,
                count in token_counts.most_common(max_keywords)]
    return set(keywords)


def rake_extract_keywords(
    text_chunk: str,
    max_keywords: Optional[int] = None,
    expand_with_subtokens: bool = True,
) -> Set[str]:
    """Extract keywords with RAKE."""
    try:
        import nltk
    except ImportError:
        raise ImportError("Please install nltk: `pip install nltk`")
    try:
        from rake_nltk import Rake
    except ImportError:
        raise ImportError("Please install rake_nltk: `pip install rake_nltk`")

    r = Rake(
        sentence_tokenizer=nltk.tokenize.sent_tokenize,
        word_tokenizer=nltk.tokenize.wordpunct_tokenize,
    )
    r.extract_keywords_from_text(text_chunk)
    keywords = r.get_ranked_phrases()[:max_keywords]
    if expand_with_subtokens:
        return set(expand_tokens_with_subtokens(keywords))
    else:
        return set(keywords)


def extract_keywords_given_response(
    response: str, lowercase: bool = True, start_token: str = ""
) -> Set[str]:
    """
    Extract keywords given the GPT-generated response.

    Used by keyword table indices.
    Parses <start_token>: <word1>, <word2>, ... into [word1, word2, ...]
    Raises exception if response doesn't start with <start_token>
    """
    results = []
    response = response.strip()  # Strip newlines from responses.

    if response.startswith(start_token):
        response = response[len(start_token):]

    keywords = response.split(",")
    for k in keywords:
        rk = k
        if lowercase:
            rk = rk.lower()
        results.append(rk.strip())

    # if keyword consists of multiple words, split into subwords
    # (removing stopwords)
    return expand_tokens_with_subtokens(set(results))


__all__ = [
    "simple_extract_keywords",
    "rake_extract_keywords",
    "extract_keywords_given_response",
]

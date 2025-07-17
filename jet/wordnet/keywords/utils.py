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

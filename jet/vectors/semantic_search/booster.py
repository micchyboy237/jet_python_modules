import numpy as np
from typing import List
from jet.vectors.semantic_search.search_types import Match


from typing import List
import numpy as np
from .search_types import Match


def boost_ngram_score(matches: List[Match], base_score: float) -> float:
    """Boost the score based on the length of the longest n-gram match."""
    if not matches:
        return base_score

    # Use the longest match for boosting
    max_length = max((m["end_idx"] - m["start_idx"]) for m in matches)

    # Adjusted boost factor: more aggressive scaling for longer matches
    boost_factor = 1.0 + 1.5 * (np.log1p(max_length) / np.log1p(100))
    boosted_score = base_score * boost_factor

    return boosted_score

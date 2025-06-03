from typing import List, Dict, TypedDict, Optional
from collections import Counter
from jet.features.nlp_utils.word_counts import get_word_counts_lemmatized
from jet.features.nlp_utils.word_sentence_counts import get_word_sentence_combination_counts
from jet.features.nlp_utils.nlp_types import WordOccurrence, Matched


class DocumentResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    matched: Dict[str, int]


class DocumentSummary(TypedDict):
    results: List[DocumentResult]
    matched: Dict[str, int]


def get_document_summary(
    texts: List[str],
    queries: List[str],
    min_count: int = 1,
    as_score: bool = True
) -> DocumentSummary:
    """
    Generate a summary of document results based on query terms.

    Args:
        texts: List of document texts to process.
        queries: List of query terms to match against documents.
        min_count: Minimum occurrence count for terms (default: 1).
        as_score: If True, use normalized scores; if False, use raw counts (default: True).

    Returns:
        DocumentSummary containing ranked document results and total matched term counts.
    """
    # Get lemmatized word counts for all texts
    word_counts_list = get_word_counts_lemmatized(
        texts,
        min_count=min_count,
        as_score=as_score
    )

    # Get word sentence combination counts for queries
    query_counts = get_word_sentence_combination_counts(
        ' '.join(queries),
        n=None,
        min_count=1,
        in_sequence=False,
        show_progress=True
    )

    # Extract query terms
    query_terms = set()
    for matched in query_counts:
        for ngram in matched['ngrams']:
            if len(ngram['ngram']) == 1:  # Single words only
                query_terms.add(ngram['ngram'][0])

    results: List[DocumentResult] = []
    total_matched: Dict[str, int] = Counter()

    # Process each document
    for doc_index, (text, word_counts) in enumerate(zip(texts, word_counts_list)):
        matched_counts: Dict[str, int] = {}
        total_score = 0.0
        total_tokens = sum(
            sum(occ['count'] for occ in occ_list)
            if not as_score else
            sum(occ['score'] for occ in occ_list)
            for occ_list in word_counts.values()
        )

        # Match query terms
        for term in query_terms:
            if term in word_counts:
                count = (
                    sum(occ['count'] for occ in word_counts[term])
                    if not as_score else
                    sum(occ['score'] for occ in word_counts[term])
                )
                matched_counts[term] = count
                total_score += count
                total_matched[term] += count

        # Normalize score to 0-100 range
        score = (total_score / max(total_tokens, 1)) * \
            100 if as_score else total_score

        results.append({
            'id': f'doc_{doc_index}',
            'rank': 0,  # Will be updated after sorting
            'doc_index': doc_index,
            'score': score,
            'text': text[:1000],  # Truncate for summary
            'tokens': int(total_tokens),
            'matched': matched_counts
        })

    # Sort results by score and assign ranks
    results.sort(key=lambda x: x['score'], reverse=True)
    for rank, result in enumerate(results, 1):
        result['rank'] = rank

    return {
        'results': results,
        'matched': dict(total_matched)
    }

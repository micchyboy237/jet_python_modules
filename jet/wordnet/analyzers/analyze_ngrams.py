from jet.logger.timer import time_it
from jet.wordnet.histogram import TextAnalysis
from jet.wordnet.similarity import filter_different_texts
from tqdm import tqdm

from typing import Any, Dict, List, Optional


@time_it
def limit_ngram_occurrences(high_ngram_tl_texts: List[Dict[str, float]], min_score: float = 0.05, texts_dict: Dict = None) -> List[Dict[str, float]]:
    high_ngram_tl_texts = [
        item for item in high_ngram_tl_texts if item['score'] >= min_score]
    if not high_ngram_tl_texts:
        print("No texts with TF-IDF score above threshold")
        return []

    high_ngram_tl_texts = sorted(
        high_ngram_tl_texts,
        key=lambda x: (x['score'], texts_dict[x['text']]['rating']
                       * 0.2 if texts_dict else 0),  # Weight rating higher
        reverse=True
    )
    high_ngram_tl_texts_dict = {item['text']
        : item for item in high_ngram_tl_texts}

    grouped_texts_by_ngram = {}
    for item in high_ngram_tl_texts:
        ngram = item['ngram']
        if ngram not in grouped_texts_by_ngram:
            grouped_texts_by_ngram[ngram] = []
        grouped_texts_by_ngram[ngram].append(item['text'])

    limited_texts = []
    for ngram, texts in grouped_texts_by_ngram.items():
        print(f"Processing n-gram '{ngram}' with {len(texts)} texts")
        diverse_texts = filter_different_texts(texts, threshold=0.5)
        if diverse_texts:
            sorted_texts = sorted(
                [high_ngram_tl_texts_dict[text] for text in diverse_texts],
                key=lambda x: (x['score'], texts_dict[x['text']]
                               ['rating'] * 0.2 if texts_dict else 0),
                reverse=True
            )[:2]
            for top_text in sorted_texts:
                limited_texts.append(top_text['text'])
                print(
                    f"Selected text for '{ngram}': '{top_text['text'][:30]}...'")

    return [high_ngram_tl_texts_dict[text] for text in set(limited_texts)]


def analyze_ngrams(
    texts: List[str],
    texts_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    min_tfidf: float = 0.05,
    ngram_ranges: List[tuple[int, int]] = [(1, 3)],
    stop_ngrams: Optional[set[str]] = None,
    top_n: int = 100,
    top_k_texts: int = 4,
) -> List[Dict[str, Any]]:
    """Generic n-gram analyzer with configurable TF-IDF and n-gram parameters."""
    # Default fallback: if no dict, simulate neutral metadata
    texts_dict = texts_dict or {text: {"rating": 0} for text in texts}

    ta = TextAnalysis(texts)
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=ngram_ranges,
        top_n=top_n,
    )

    stop_ngrams = stop_ngrams or set()
    results = [
        item for sublist in most_any_results for item in sublist['results']
        if 'tfidf' in item
    ]

    # ---- Filter results by normalized TF-IDF ----
    if not results:
        print("No n-grams found with TF-IDF values.")
        return []

    max_score = max((result.get('tfidf', 0.0) for result in results), default=0.0)
    if max_score <= 0:
        print("All TF-IDF scores are zero. Skipping normalization and returning empty list.")
        return []

    results = [
        item for item in results
        if (item.get('tfidf', 0.0) / max_score) > min_tfidf
        and item['ngram'] not in stop_ngrams
    ]

    if not results:
        print("No n-grams meet TF-IDF and stopword criteria.")
        return []

    ngram_tfidf_dict = {r['ngram']: r['tfidf'] for r in results}

    high_ngram_tl_texts = []
    pbar = tqdm(texts, desc="Scanning texts")
    for text in pbar:
        matched_ngrams = [
            {"ngram": n, "tfidf": s}
            for n, s in ngram_tfidf_dict.items()
            if n.lower() in text.lower()
        ]
        if matched_ngrams:
            max_tfidf = max(m['tfidf'] for m in matched_ngrams)
            high_ngram_tl_texts.append({
                "ngram": matched_ngrams[0]["ngram"],
                "score": max_tfidf,
                "text": text,
                "matched_ngrams": matched_ngrams
            })

    if not high_ngram_tl_texts:
        print("No texts with matching n-grams.")
        return []

    limited_tl_texts = limit_ngram_occurrences(
        high_ngram_tl_texts, min_score=min_tfidf, texts_dict=texts_dict
    )

    filtered_tl_texts = [
        {
            **texts_dict[item['text']],
            "text": item['text'],
            "matched_ngrams": item['matched_ngrams'],
            "max_tfidf": item['score']
        } for item in limited_tl_texts
    ]

    filtered_tl_texts = sorted(
        filtered_tl_texts,
        key=lambda x: (x['max_tfidf'], x.get('rating', 0) * 0.2),
        reverse=True
    )[:top_k_texts]

    return filtered_tl_texts


def generate_histograms(data):
    from jet.wordnet.histogram import generate_histograms as base_generate_histograms
    return base_generate_histograms(data)

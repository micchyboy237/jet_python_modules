import os
from jet.file.utils import save_data
from jet.logger.timer import time_it
from jet.wordnet.histogram import TextAnalysis
# from jet.wordnet.similarity import filter_different_texts
from jet.wordnet.similarity import filter_different_texts
from tqdm import tqdm

from difflib import SequenceMatcher
from typing import Dict, List, Callable


@time_it
def limit_ngram_occurrences(high_ngram_tl_texts: List[Dict[str, float]], min_score: float = 0.03, texts_dict: Dict = None) -> List[Dict[str, float]]:
    high_ngram_tl_texts = [
        item for item in high_ngram_tl_texts if item['score'] >= min_score]
    if not high_ngram_tl_texts:
        print("No texts with TF-IDF score above threshold")
        return []

    high_ngram_tl_texts = sorted(
        high_ngram_tl_texts,
        key=lambda x: (x['score'], texts_dict[x['text']]
                       ['rating'] if texts_dict else 0),
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
            # Select up to 2 texts per n-gram, prioritizing score and rating
            sorted_texts = sorted(
                [high_ngram_tl_texts_dict[text] for text in diverse_texts],
                key=lambda x: (x['score'], texts_dict[x['text']]
                               ['rating'] if texts_dict else 0),
                reverse=True
            )[:2]
            for top_text in sorted_texts:
                limited_texts.append(top_text['text'])
                print(
                    f"Selected text for '{ngram}': '{top_text['text'][:30]}...'")

    return [high_ngram_tl_texts_dict[text] for text in set(limited_texts)]


def analyze_ngrams(texts, texts_dict, min_tfidf=0.03):
    ta = TextAnalysis(texts)
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(1, 3)],
        top_n=100,
    )
    most_any_results_text_dict = {}
    stop_ngrams = {
        "but the", "and a", "life and", "the battery", "battery life and",
        "life and a", "but the battery", "battery life and a", "the phone",
        "the camera", "during heavy", "is fast", "is stunning", "when recording",
        "are slow", "with a"
    }
    results = [
        item for sublist in most_any_results for item in sublist['results']
        if 'tfidf' in item
    ]
    print(f"Found {len(results)} n-grams (before min_tfidf filter):")
    for result in results:
        print(f"  {result['ngram']}: {result['tfidf']:.4f}")

    # Normalize TF-IDF scores
    max_score = max((result['tfidf'] for result in results), default=1)
    results = [
        item for item in results
        if item['tfidf'] / max_score > min_tfidf and item['ngram'] not in stop_ngrams
    ]
    print(
        f"Found {len(results)} n-grams with normalized TF-IDF > {min_tfidf} after stopword filter:")
    for result in results:
        most_any_results_text_dict[result['ngram']] = result['tfidf']
        print(f"  {result['ngram']}: {result['tfidf']:.4f}")

    if not results:
        print("No n-grams meet the TF-IDF and stopword criteria. Returning empty list.")
        return []

    high_ngram_tl_texts = []
    pbar = tqdm(texts, desc="Processing texts")
    for text in pbar:
        matched_ngrams = []
        max_tfidf = 0
        for ngram, score in most_any_results_text_dict.items():
            if ngram.lower() in text.lower():
                matched_ngrams.append({"ngram": ngram, "tfidf": score})
                max_tfidf = max(max_tfidf, score)
        if matched_ngrams:
            high_ngram_tl_texts.append({
                "ngram": matched_ngrams[0]["ngram"],
                "score": max_tfidf,
                "text": text,
                "matched_ngrams": matched_ngrams
            })
            pbar.set_description(
                f"High-ngram texts: {len(high_ngram_tl_texts)}")

    print(f"Texts with high-scoring n-grams: {len(high_ngram_tl_texts)}")
    if not high_ngram_tl_texts:
        print("No texts contain high-scoring n-grams. Returning empty list.")
        return []

    limited_tl_texts = limit_ngram_occurrences(
        high_ngram_tl_texts, min_score=min_tfidf, texts_dict=texts_dict)
    filtered_tl_texts = [
        {
            **texts_dict[item['text']],
            "text": item['text'],
            "matched_ngrams": item['matched_ngrams'],
            "max_tfidf": item['score']
        } for item in limited_tl_texts
    ]
    print(f"Final filtered texts: {len(filtered_tl_texts)}")

    filtered_tl_texts = sorted(
        filtered_tl_texts,
        key=lambda x: (x['max_tfidf'], x['rating']),
        reverse=True
    )[:5]

    return filtered_tl_texts


def generate_histograms(data):
    ta = TextAnalysis(data)

    most_start_results = ta.generate_histogram(
        is_top=True,
        from_start=True,
        ngram_ranges=[(1, 1), (2, 2)],
        top_n=100,
    )
    least_start_results = ta.generate_histogram(
        is_top=False,
        from_start=True,
        ngram_ranges=[(1, 1), (2, 2)],
        top_n=100,
    )
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(2, 2), (3, 5)],
        top_n=100,
    )
    least_any_results = ta.generate_histogram(
        is_top=False,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(2, 2), (3, 5)],
        top_n=100,
    )

    return {
        'most_common_start': most_start_results,
        'least_common_start': least_start_results,
        'most_common_any': most_any_results,
        'least_common_any': least_any_results,
    }

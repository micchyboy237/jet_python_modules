import re
from typing import List, Dict, Optional, TypedDict, Union

from tqdm import tqdm
from jet.wordnet.words import get_words
from jet.logger import time_it
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

class TopDocumentResult(TypedDict):
    doc_idx: int
    doc: str
    score_tfidf: float
    score_collocation: float
    score_combined: float

class TextAnalysis:
    def __init__(self, data, eos_token_str='<|eos|>', include_keys=[], exclude_keys=[]):
        self.data = data
        self.eos_token_str = eos_token_str
        self.include_keys = include_keys
        self.exclude_keys = exclude_keys
        self.tokens_collocation_any = self.tokenize_and_clean_for_collocation_any(
            data)
        self.tokens_tfidf = self.tokenize_and_clean_for_tfidf()

    def tokenize_and_clean_for_collocation_start(self, n) -> List[str]:
        all_tokens = []
        for item in self.data:
            if isinstance(item, str):
                text = item
            else:
                keys_to_use = self.include_keys if self.include_keys else item.keys()
                text = "\n".join(
                    str(item[key]) for key in keys_to_use
                    if key != "id"
                    and (not self.include_keys or key in self.include_keys)
                    and (not self.exclude_keys or key not in self.exclude_keys)
                )
            starting_tokens = get_words(text)
            starting_tokens = starting_tokens[:n]
            all_tokens.extend(starting_tokens)
        return all_tokens

    def tokenize_and_clean_for_collocation_any(self, data: List[Union[str, Dict]]) -> List[str]:
        all_tokens = []
        for item in data:
            if isinstance(item, str):
                text = item
            else:
                keys_to_use = self.include_keys if self.include_keys else item.keys()
                text = "\n".join(
                    str(item[key]) for key in keys_to_use
                    if key != "id" and key not in self.exclude_keys
                )
            tokens = get_words(text)
            all_tokens.extend(tokens)
        return all_tokens

    def tokenize_and_clean_for_tfidf(self) -> List[str]:
        """Tokenize TF-IDF texts with respect to newline and EOS boundaries."""
        all_texts = []
        for item in self.data:
            if isinstance(item, str):
                text = item
            else:
                keys_to_use = self.include_keys if self.include_keys else item.keys()
                text = "\n".join(
                    str(item[key]) for key in keys_to_use
                    if key != "id" and key not in self.exclude_keys
                )

            # Replace newline or EOS markers with sentence boundary token
            text = re.sub(rf"({self.eos_token_str}|\n+)", " <eos> ", text.strip().lower())
            all_texts.append(text)
        return all_texts

    @time_it
    def perform_tfidf_analysis(self, ngram_range):
        """Perform TF-IDF respecting word and sentence boundaries."""
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            stop_words='english',
            analyzer='word',
            preprocessor=lambda x: re.sub(r'\s+', ' ', x.strip())  # collapse multiple spaces only
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.tokens_tfidf)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        top_ngrams = sorted(
            zip(feature_names, tfidf_matrix.sum(axis=0).A1),
            key=lambda x: x[1],
            reverse=True
        )
        return top_ngrams

    def _split_on_newlines(self, text: str) -> List[str]:
        """Split text into segments by newlines, preserving boundaries."""
        return [seg.strip() for seg in re.split(r'\n+', text) if seg.strip()]

    def perform_dynamic_collocation_analysis(self, ngram_range, separated_texts):
        """Perform collocation analysis with newline/EOS boundary awareness."""
        all_collocations = {}
        for text in self.data:
            if isinstance(text, str):
                text_content = text
            else:
                keys_to_use = self.include_keys if self.include_keys else text.keys()
                text_content = "\n".join(
                    str(text[key]) for key in keys_to_use
                    if key != "id" and key not in self.exclude_keys
                )

            # Split by newline and EOS tokens; treat each as a separate boundary
            segments = self._split_on_newlines(
                re.sub(rf"{self.eos_token_str}", "\n", text_content)
            )

            for segment in segments:
                tokens = get_words(segment)
                if not tokens:
                    continue
                for n in range(ngram_range[0], ngram_range[1] + 1):
                    for ngram in nltk.ngrams(tokens, n):
                        ngram_text = ' '.join(ngram)
                        all_collocations[ngram_text] = all_collocations.get(ngram_text, 0) + 1

        return all_collocations

    def _compute_ngram_doc_counts(self, ngrams: List[str]) -> Dict[str, int]:
        """Count how many docs each ngram appears in (each doc counted once),
        ignoring punctuation on word edges (e.g., 'Tech:' == 'Tech')."""
        doc_counts: Dict[str, int] = {ngram: 0 for ngram in ngrams}
        for doc in self.tokens_tfidf:
            # Normalize: lowercase and remove excessive whitespace
            doc_lower = re.sub(r'\s+', ' ', doc.lower())

            for ngram in ngrams:
                # Clean and normalize the ngram for safe regex use
                clean_ngram = re.escape(ngram.lower())

                # Allow optional punctuation before/after each token in the ngram
                # Example: match "Tech:" or "(Tech)" for ngram "Tech"
                pattern = r'(?<!\w)[^\w\s]*' + clean_ngram + r'[^\w\s]*(?!\w)'

                if re.search(pattern, doc_lower):
                    doc_counts[ngram] += 1

        return doc_counts

    def format_results(
        self,
        scored_collocations,
        top_ngrams,
        is_top: bool = True,
        apply_tfidf: bool = True,
        from_start: bool = False,
        top_n: Optional[int] = None,
        remove_stopwords: bool = False
    ):
        stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        def has_stopword_at_edges(ngram: str) -> bool:
            tokens = ngram.lower().split()
            if not tokens:
                return False
            return tokens[0] in stop_words or tokens[-1] in stop_words

        # --- Collocations ---
        collocations_results = sorted(
            scored_collocations.items(), key=lambda x: x[1], reverse=is_top
        )
        collocations_results = [
            {'ngram': ngram, 'collocations': score, 'n': len(ngram.split())}
            for ngram, score in collocations_results
            if not (remove_stopwords and has_stopword_at_edges(ngram))
        ]
        if is_top:
            collocations_results = [
                result for result in collocations_results if result['collocations'] > 1
            ]
        if top_n:
            collocations_results = collocations_results[:top_n]

        results = {
            'scope': 'start' if from_start else 'any',
            'results': collocations_results,
        }

        # --- TF-IDF ---
        if apply_tfidf:
            tfidf_results = sorted(
                top_ngrams, key=lambda x: x[1], reverse=is_top
            )
            tfidf_results = [
                {'ngram': ngram, 'tfidf': score, 'n': len(ngram.split())}
                for ngram, score in tfidf_results
                if not (remove_stopwords and has_stopword_at_edges(ngram))
            ]

            if top_n:
                if 0 < top_n < 1:
                    tfidf_results = [
                        result for result in tfidf_results if result['tfidf'] > top_n
                    ]
                else:
                    tfidf_results = tfidf_results[:top_n]

            # Merge TF-IDF scores
            for i, result in enumerate(results['results']):
                ngram = result['ngram']
                tfidf_score = next(
                    (x['tfidf'] for x in tfidf_results if x['ngram'] == ngram), 0
                )
                results['results'][i]['tfidf'] = tfidf_score

        # --- Document counts ---
        all_ngrams = [r['ngram'] for r in results['results']]
        ngram_doc_counts = self._compute_ngram_doc_counts(all_ngrams)
        for r in results['results']:
            r['docs'] = ngram_doc_counts.get(r['ngram'], 0)

        return results

    @time_it
    def filter_longest_ngrams(self, results: List[Dict[str, float]], key: str) -> List[Dict[str, float]]:
        results.sort(key=lambda x: (-len(x['ngram']), -x[key]))
        filtered_results = []
        seen_sub_ngrams = set()
        for current in results:
            if current['ngram'] not in seen_sub_ngrams:
                filtered_results.append(current)
                for other in results:
                    if other['ngram'] != current['ngram'] and other['ngram'] in current['ngram']:
                        seen_sub_ngrams.add(other['ngram'])
        return filtered_results

    def filter_top_documents_by_tfidf_and_collocations(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        weight_tfidf: float = 0.5,
        weight_collocation: float = 0.5,
        top_n: int = 10,
        show_progress: bool = False,
    ) -> List[TopDocumentResult]:
        """
        Rank and filter documents based on combined TF-IDF and collocation relevance.

        Args:
            ngram_range: Tuple defining n-gram range for analysis.
            weight_tfidf: Weight for TF-IDF score contribution.
            weight_collocation: Weight for collocation score contribution.
            top_n: Number of top documents to return.
            show_progress: Whether to display a tqdm progress bar.
        """
        # --- Early exit if no valid documents ---
        if not self.tokens_tfidf or all(not str(t).strip() for t in self.tokens_tfidf):
            return []

        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            stop_words='english',
            analyzer='word'
        )

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.tokens_tfidf)
        except ValueError:
            # Handle case: all docs are stopwords or empty
            return []

        tfidf_scores = tfidf_matrix.toarray()
        collocations = self.perform_dynamic_collocation_analysis(ngram_range, None)

        # Normalize collocation frequencies
        max_colloc = max(collocations.values(), default=1)
        colloc_scores_norm = {ng: sc / max_colloc for ng, sc in collocations.items()}

        iterator = tqdm(
            enumerate(self.tokens_tfidf),
            total=len(self.tokens_tfidf),
            disable=not show_progress,
            desc="Scoring documents",
            unit="doc",
        )

        results: List[TopDocumentResult] = []
        for i, doc_clean in iterator:
            original_doc = self.data[i] if isinstance(self.data[i], str) else str(self.data[i])

            score_tfidf = float(tfidf_scores[i].sum())
            score_collocation = sum(
                colloc_scores_norm[ngram]
                for ngram in colloc_scores_norm
                if re.search(rf'\b{re.escape(ngram)}\b', doc_clean)
            )
            score_combined = (weight_tfidf * score_tfidf) + (weight_collocation * score_collocation)

            results.append({
                "doc_idx": i,
                "doc": original_doc,
                "score_tfidf": score_tfidf,
                "score_collocation": score_collocation,
                "score_combined": score_combined,
            })

        return sorted(results, key=lambda x: x["score_combined"], reverse=True)[:top_n]

    def generate_histogram(self, from_start=False, apply_tfidf=False, ngram_ranges=None, is_top=True, top_n=None, remove_stopwords=False):
        ngram_ranges = ngram_ranges or [(1, 2)]
        all_results = []
        for ngram_range in ngram_ranges:
            print(f"\nAnalyzing for ngram range: {ngram_range}")
            ngram_range_str = "{}-{}".format(*ngram_range)
            print("Performing dynamic collocation analysis...")
            collocation_results = self.perform_dynamic_collocation_analysis(
                # Pass None as separated_texts, handled in perform_dynamic_collocation_analysis
                ngram_range, None)
            top_ngrams = self.perform_tfidf_analysis(
                ngram_range) if apply_tfidf else None
            results_dict = self.format_results(
                collocation_results, top_ngrams, is_top=is_top, apply_tfidf=apply_tfidf, from_start=from_start, top_n=top_n, remove_stopwords=remove_stopwords)
            result_entry = {
                'ngram_range': ngram_range_str,
                **results_dict,
            }
            all_results.append(result_entry)
            print(
                f"Processed {'start' if is_top else 'any'} ngram range {ngram_range_str}")
        return all_results

def generate_histograms(data):
    ta = TextAnalysis(data)

    most_start_results = ta.generate_histogram(
        is_top=True,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=200,
    )
    least_start_results = ta.generate_histogram(
        is_top=False,
        from_start=True,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 1), (2, 3)],
        top_n=200,
    )
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=200,
    )
    least_any_results = ta.generate_histogram(
        is_top=False,
        from_start=False,
        apply_tfidf=True,
        remove_stopwords=True,
        ngram_ranges=[(1, 3), (4, 6)],
        top_n=200,
    )

    return {
        'most_common_start': most_start_results,
        'least_common_start': least_start_results,
        'most_common_any': most_any_results,
        'least_common_any': least_any_results,
    }

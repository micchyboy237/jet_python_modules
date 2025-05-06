from typing import List, Dict, Union
from jet.wordnet.words import get_words
from jet.file.utils import load_data, load_data_from_directories
from jet.logger import time_it
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
import nltk
import os
import json


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
        all_texts = []
        for item in self.data:
            if isinstance(item, str):
                text = item
            else:
                keys_to_use = self.include_keys if self.include_keys else item.keys()
                text = "\n".join(str(item[key]) for key in keys_to_use
                                 if key != "id" and key not in self.exclude_keys)
            all_texts.append(text.lower())
        return all_texts

    @time_it
    def perform_tfidf_analysis(self, ngram_range):
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            stop_words='english'
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.tokens_tfidf)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        top_ngrams = sorted(
            list(zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0])),
            key=lambda x: x[1],
            reverse=True
        )
        return top_ngrams

    @time_it
    def perform_dynamic_collocation_analysis(self, ngram_range, separated_texts):
        all_collocations = {}
        separated_texts = [text.replace(self.eos_token_str, '')
                           for text in separated_texts]
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngram_collocations = nltk.ngrams(separated_texts, n)
            freq_dist = FreqDist(ngram_collocations)
            for ngram, freq in freq_dist.items():
                ngram_text = ' '.join(ngram)
                if ngram_text not in all_collocations:
                    all_collocations[ngram_text] = freq
        return all_collocations

    def format_results(self, scored_collocations, top_ngrams, is_top=True, apply_tfidf=True, from_start=False, top_n=None):
        collocations_results = sorted(
            scored_collocations.items(), key=lambda x: x[1], reverse=is_top)
        collocations_results = [{'ngram': ngram, 'collocations': score}
                                for ngram, score in collocations_results]
        if is_top:
            collocations_results = [
                result for result in collocations_results if result['collocations'] > 1]
        if top_n:
            collocations_results = collocations_results[:top_n]

        results = {
            'scope': 'start' if from_start else 'any',
            'results': collocations_results,
        }

        if apply_tfidf:
            tfidf_results = sorted(
                top_ngrams, key=lambda x: x[1], reverse=is_top)
            tfidf_results = [{'ngram': ngram, 'tfidf': score}
                             for ngram, score in tfidf_results]
            if top_n:
                if top_n > 0 and top_n < 1:
                    tfidf_results = [
                        result for result in tfidf_results if result['tfidf'] > top_n]
                else:
                    tfidf_results = tfidf_results[:top_n]

            for i, result in enumerate(results['results']):
                ngram = result['ngram']
                tfidf_score = next(
                    (x['tfidf'] for x in tfidf_results if x['ngram'] == ngram), 0)
                results['results'][i]['tfidf'] = tfidf_score

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

    def generate_histogram(self, from_start, apply_tfidf=False, ngram_ranges=None, is_top=True, top_n=None):
        apply_tfidf = not from_start
        ngram_ranges = ngram_ranges or [(1, 2)]
        all_results = []
        for ngram_range in ngram_ranges:
            print(f"\nAnalyzing for ngram range: {ngram_range}")
            ngram_range_str = "{}-{}".format(*ngram_range)
            highest_ngram_max = ngram_range[1]
            separated_texts = self.tokenize_and_clean_for_collocation_start(
                n=highest_ngram_max) if from_start else self.tokens_collocation_any
            print("Performing dynamic collocation analysis...")
            collocation_results = self.perform_dynamic_collocation_analysis(
                ngram_range, separated_texts)
            top_ngrams = self.perform_tfidf_analysis(
                ngram_range) if apply_tfidf else None
            results_dict = self.format_results(
                collocation_results, top_ngrams, is_top=is_top, apply_tfidf=apply_tfidf, from_start=from_start, top_n=top_n)
            result_entry = {
                'ngram_range': ngram_range_str,
                **results_dict,
            }
            all_results.append(result_entry)
            print(
                f"Processed {'start' if is_top else 'any'} ngram range {ngram_range_str}")
        return all_results

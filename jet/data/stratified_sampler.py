import math
from typing import List, Optional, TypedDict, Tuple, Union
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from jet.data.stratified_sampler_helpers.data_labeling import DataLabeler
from jet.data.stratified_sampler_helpers.sampler_types import ProcessedData, ProcessedDataString, StratifiedData
from jet.logger import logger, time_it
from jet.wordnet.n_grams import get_ngram_weight, get_ngrams, filter_and_sort_sentences_by_ngrams
from jet.wordnet.words import get_unique_words, get_words
from tqdm import tqdm
import numpy as np
import itertools


class StratifiedSampler:
    def __init__(self, data: List[ProcessedData] | List[str] | List[ProcessedDataString], num_samples: Optional[float | int] = None):
        if not data:
            raise ValueError("Input data cannot be empty")
        if isinstance(data[0], dict):
            if all(key in data[0] for key in ['source', 'category_values']):
                str_data = [item['source'] for item in data]
                # Validate category_values types
                for item in data:
                    if not all(isinstance(val, (str, int, float, bool)) for val in item['category_values']):
                        raise ValueError(
                            "category_values must contain only str, int, float, or bool")
                self.data = data
            else:
                raise ValueError(
                    "Dictionary input must contain 'source' and 'category_values' keys")
        else:
            str_data = data
            self.data = data
        unique_words = get_unique_words(str_data)

        if isinstance(num_samples, float) and 0.0 < num_samples < 1.0:
            final_num_samples = int(num_samples * len(data))
        elif isinstance(num_samples, int) and num_samples > 0:
            final_num_samples = min(num_samples, len(data))
        else:
            # Dynamically calculate num_samples dynamically if None
            final_num_samples = self._calculate_optimal_num_samples(
                data, str_data)
        self.num_samples = final_num_samples

    def _calculate_optimal_num_samples(self, data: List[ProcessedData] | List[str] | List[ProcessedDataString], str_data: List[str]) -> int:
        """Calculate optimal num_samples based on dataset size and diversity."""
        total_size = len(data)
        if total_size == 0:
            return 0

        # Estimate diversity based on unique category_values or n-grams
        if isinstance(data[0], dict):
            unique_categories = len(
                set(tuple(item['category_values']) for item in data))
        else:
            # For string data, use unique starting bigrams as a proxy for diversity
            n = 2
            starting_ngrams = Counter(
                " ".join(get_ngrams(s, n)[:1]) for s in str_data if get_ngrams(s, n))
            unique_categories = len(starting_ngrams) or 1

        # Heuristic: min(80% of data, sqrt(unique_categories) * log(total_size))
        max_proportion = int(0.8 * total_size)
        diversity_based = int(math.sqrt(unique_categories)
                              * math.log1p(total_size))
        optimal_samples = max(
            1, min(max_proportion, diversity_based, total_size))
        return optimal_samples

    def _calculate_optimal_top_n(self, sentences: List[str], n: int) -> int:
        """Calculate optimal top_n based on number of unique n-grams and total sentences."""
        ngrams = Counter()
        for sentence in sentences:
            ngrams.update(get_ngrams(sentence, n))
        unique_ngrams = len(ngrams)
        if unique_ngrams == 0:
            return 2  # Default fallback
        # Aim for a top_n that balances diversity and coverage
        total_sentences = len(sentences)
        top_n = max(1, min(total_sentences, int(math.sqrt(unique_ngrams))))
        return top_n

    @time_it
    def filter_strings(self, n: int = 2, top_n: Optional[int] = None) -> List[str]:
        sentences = self.data if isinstance(self.data[0], str) else [
            item['source'] for item in self.data]

        # Dynamically calculate top_n if None
        if top_n is None:
            top_n = self._calculate_optimal_top_n(sentences, n)

        filtered_data = filter_and_sort_sentences_by_ngrams(
            sentences, n, top_n, is_start_ngrams=True)
        return filtered_data[:self.num_samples]

    @time_it
    def get_samples(self) -> Union[List[str], List[StratifiedData]]:
        if isinstance(self.data[0], str):
            filtered_sentences = self.filter_strings()
            return filtered_sentences

        labels = [item['category_values'] for item in self.data]
        has_target = isinstance(
            self.data[0], dict) and 'target' in self.data[0] and 'score' in self.data[0]
        if has_target:
            score_map = {(item['source'], item['target'])                         : item['score'] for item in self.data}
            features_targets = [(item['source'], item['target'])
                                for item in self.data]
        else:
            score_map = {(item['source'], None): None for item in self.data}
            features_targets = [(item['source'], None) for item in self.data]
        # Calculate proportional sample sizes per category
        category_counts = Counter(tuple(lbl) for lbl in labels)
        total_data = len(self.data)
        samples_per_category = {
            cat: max(1, int(self.num_samples * count / total_data))
            for cat, count in category_counts.items()
        }
        # Adjust to strictly match num_samples
        total_assigned = sum(samples_per_category.values())
        if total_assigned > self.num_samples:
            excess = total_assigned - self.num_samples
            for cat in sorted(samples_per_category, key=samples_per_category.get, reverse=True):
                while excess > 0 and samples_per_category[cat] > 1:
                    samples_per_category[cat] -= 1
                    excess -= 1
        elif total_assigned < self.num_samples:
            deficit = self.num_samples - total_assigned
            for cat in sorted(samples_per_category, key=samples_per_category.get):
                if deficit == 0:
                    break
                samples_per_category[cat] += 1
                deficit -= 1
        # Sample from each category
        selected_samples = []
        selected_labels = []
        for cat, n_samples in samples_per_category.items():
            cat_indices = [i for i, lbl in enumerate(
                labels) if tuple(lbl) == cat]
            if n_samples > len(cat_indices):
                n_samples = len(cat_indices)
            try:
                sampled_indices, _ = train_test_split(
                    cat_indices, train_size=n_samples, random_state=42
                )
            except ValueError:
                sampled_indices = cat_indices[:n_samples]
            for idx in sampled_indices:
                selected_samples.append(features_targets[idx])
                selected_labels.append(labels[idx])
        # Ensure exact num_samples
        if len(selected_samples) > self.num_samples:
            selected_samples = selected_samples[:self.num_samples]
            selected_labels = selected_labels[:self.num_samples]
        stratified_samples = [
            StratifiedData(source=ft[0], target=ft[1],
                           score=score_map[ft], category_values=lbl)
            for ft, lbl in zip(selected_samples, selected_labels)
        ]
        # Precompute n-grams for sorting
        n = 2  # Consistent with default in filter_and_sort_sentences_by_ngrams
        all_ngrams = Counter()
        sentence_ngrams_dict = {}
        sentences = [sample['source'] for sample in stratified_samples]
        for sentence in sentences:
            ngram_list = get_ngrams(sentence, n)
            all_ngrams.update(ngram_list)
            sentence_ngrams_dict[sentence] = ngram_list
        # Sort samples based on n-gram weights
        sorted_samples = []
        for _ in range(len(stratified_samples)):
            if sorted_samples:
                previous_ngrams = set(get_ngrams(
                    sorted_samples[-1]['source'], n))
            else:
                previous_ngrams = set()
            stratified_samples.sort(key=lambda sample: (
                get_ngram_weight(
                    all_ngrams, sentence_ngrams_dict[sample['source']], previous_ngrams),
                sample['source']
            ))
            sorted_samples.append(stratified_samples.pop(0))
        return sorted_samples

    @time_it
    def get_unique_strings(self) -> List[str]:
        data_with_labels = self.load_data_with_labels()
        features_targets = [item['source'] for item in data_with_labels]
        labels = [item['category_values'] for item in data_with_labels]
        try:
            features_targets_sample, _, labels_sample, _ = train_test_split(
                features_targets, labels, train_size=self.num_samples, stratify=labels
            )
        except ValueError:
            features_targets_sample, _, labels_sample, _ = train_test_split(
                features_targets, labels, train_size=self.num_samples
            )
        return features_targets_sample

    @time_it
    def load_data_with_labels(self, max_q: int = 2) -> List[ProcessedDataString]:
        data = self.data if isinstance(self.data[0], str) else [
            item['source'] for item in self.data]
        labeler = DataLabeler(data, max_q)
        return labeler.label_data()

    @time_it
    def split_train_test_val(self, train_ratio: float = 0.6, test_ratio: float = 0.2) -> Tuple[List[ProcessedData | ProcessedDataString], List[ProcessedData | ProcessedDataString], List[ProcessedData | ProcessedDataString]]:
        """Split data into train, test, and validation sets with stratification."""
        if not isinstance(self.data[0], dict):
            raise ValueError(
                "split_train_test_val requires ProcessedData or ProcessedDataString input")
        if not (0 < train_ratio < 1 and 0 < test_ratio < 1 and train_ratio + test_ratio < 1):
            raise ValueError(
                "train_ratio and test_ratio must be in (0, 1) and sum to less than 1")

        labels = [item['category_values'] for item in self.data]
        has_target = 'target' in self.data[0] and 'score' in self.data[0]

        if has_target:
            # Handle ProcessedData with target and score
            score_map = {(item['source'], item['target'])                         : item['score'] for item in self.data}
            features_targets = [(item['source'], item['target'])
                                for item in self.data]
        else:
            # Handle ProcessedDataString with only source
            features_targets = [item['source'] for item in self.data]

        # First split: train + validation vs. test
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                features_targets, labels, test_size=test_ratio, stratify=labels
            )
        except ValueError:
            # Fallback to non-stratified sampling
            X_temp, X_test, y_temp, y_test = train_test_split(
                features_targets, labels, test_size=test_ratio
            )

        # Second split: train vs. validation
        val_ratio = (1 - train_ratio - test_ratio) / (1 - test_ratio)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, stratify=y_temp
            )
        except ValueError:
            # Fallback to non-stratified sampling
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio
            )

        if has_target:
            # Convert to ProcessedData format
            train_data = [
                ProcessedData(source=ft[0], target=ft[1],
                              score=score_map[ft], category_values=lbl)
                for ft, lbl in zip(X_train, y_train)
            ]
            test_data = [
                ProcessedData(source=ft[0], target=ft[1],
                              score=score_map[ft], category_values=lbl)
                for ft, lbl in zip(X_test, y_test)
            ]
            val_data = [
                ProcessedData(source=ft[0], target=ft[1],
                              score=score_map[ft], category_values=lbl)
                for ft, lbl in zip(X_val, y_val)
            ]
        else:
            # Convert to ProcessedDataString format
            train_data = [
                ProcessedDataString(source=ft, category_values=lbl)
                for ft, lbl in zip(X_train, y_train)
            ]
            test_data = [
                ProcessedDataString(source=ft, category_values=lbl)
                for ft, lbl in zip(X_test, y_test)
            ]
            val_data = [
                ProcessedDataString(source=ft, category_values=lbl)
                for ft, lbl in zip(X_val, y_val)
            ]

        return train_data, test_data, val_data

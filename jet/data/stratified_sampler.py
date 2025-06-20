from typing import List, Optional, TypedDict, Tuple, Union
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from jet.logger import logger, time_it
from jet.wordnet.words import get_unique_words, get_words
from tqdm import tqdm
import numpy as np
import itertools

# Define primitive types for category_values
PrimitiveType = Union[str, int, float, bool]


class ProcessedData(TypedDict):
    source: str
    target: Optional[str]
    category_values: List[PrimitiveType]
    score: Optional[float]


class ProcessedDataString(TypedDict):
    source: str
    category_values: List[PrimitiveType]


class StratifiedData(TypedDict):
    source: str
    target: Optional[str]
    score: Optional[float]
    category_values: List[PrimitiveType]


def get_ngrams(text: str, n: int = 1) -> List[str]:
    words = get_words(text, n)
    return words


def get_ngram_weight(all_ngrams: Counter, sentence_ngrams: List[str], previous_ngrams: set) -> float:
    penalty = sum(ngram in previous_ngrams for ngram in sentence_ngrams)
    return sum(1 / all_ngrams[ngram] for ngram in sentence_ngrams if ngram in all_ngrams) + penalty


def sort_sentences(sentences: List[str], n: int) -> List[str]:
    all_ngrams = Counter()
    sentence_ngrams_dict = {}
    for sentence in tqdm(sentences, desc="Precomputing n-grams"):
        ngram_list = get_ngrams(sentence, n)
        all_ngrams.update(ngram_list)
        sentence_ngrams_dict[sentence] = ngram_list
    sorted_sentences = []
    for _ in tqdm(range(len(sentences)), desc="Sorting sentences"):
        if sorted_sentences:
            previous_ngrams = set(get_ngrams(sorted_sentences[-1], n))
        else:
            previous_ngrams = set()
        sentences.sort(key=lambda sentence: (get_ngram_weight(
            all_ngrams, sentence_ngrams_dict[sentence], previous_ngrams), sentence))
        sorted_sentences.append(sentences.pop(0))
    return sorted_sentences


def n_gram_frequency(sentence: str, n: int = 2) -> Counter:
    """Calculate the frequency of character-based n-grams in a sentence"""
    n_grams = [sentence[i:i+n] for i in range(len(sentence) - n + 1)]
    return Counter(n_grams)


def calculate_n_gram_diversity(freq: Counter) -> int:
    """Calculate diversity based on the count of unique n-grams"""
    return len(freq)


def filter_and_sort_sentences_by_ngrams(sentences: List[str], n: int = 2, top_n: int = 2, is_start_ngrams: bool = True) -> List[str]:
    sentence_ngrams = defaultdict(list)
    all_ngrams = Counter()
    for sentence in tqdm(sentences, desc="Grouping sentences"):
        ngrams_list = get_ngrams(sentence, n)
        all_ngrams.update(ngrams_list)
        if is_start_ngrams and ngrams_list:
            sentence_ngrams[" ".join(ngrams_list[:1])].append(sentence)
        elif not is_start_ngrams:
            for ngram in set(ngrams_list):
                sentence_ngrams[" ".join(ngram)].append(sentence)
    optimized_groups = {ngram: group_sentences[:top_n]
                        for ngram, group_sentences in sentence_ngrams.items()}
    flattened_sentences = list(set(
        itertools.chain.from_iterable(optimized_groups.values())))
    # Stable sort with secondary key (lexicographical order)
    sorted_sentences = sort_sentences(flattened_sentences, n)
    sorted_sentences.sort(key=lambda x: (
        get_ngram_weight(all_ngrams, get_ngrams(x, n), set()), x))
    return sorted_sentences[:top_n]


class StratifiedSampler:
    def __init__(self, data: List[ProcessedData | str | ProcessedDataString], num_samples: float | int = 0.8):
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
            raise ValueError(
                "num_samples must be a float in the range (0.0, 1.0) or a positive integer")
        self.num_samples = final_num_samples

    @time_it
    def filter_strings(self, n: int = 2, top_n: int = 2) -> List[str]:
        filtered_data = filter_and_sort_sentences_by_ngrams(
            self.data if isinstance(self.data[0], str) else [item['source'] for item in self.data], n, top_n, is_start_ngrams=True)
        return filtered_data[:self.num_samples]

    @time_it
    def get_samples(self) -> List[StratifiedData]:
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

        def calculate_ttr(sentence: str) -> int:
            words = sentence.split()
            unique_words = set(words)
            return len(unique_words)

        def calculate_ttr_class(ttr: int, ttr_quantiles: np.ndarray) -> str:
            for i, q in enumerate(ttr_quantiles):
                if ttr <= q:
                    return f'ttr_q{i+1}'
            return f'ttr_q{len(ttr_quantiles)+1}'

        def categorize_sentence_length(sentence: str, length_quantiles: np.ndarray) -> str:
            word_count = len(sentence.split())
            for i, q in enumerate(length_quantiles):
                if word_count <= q:
                    return f'q{i+1}'
            return f'q{len(length_quantiles)+1}'

        def categorize_n_gram_diversity(diversity: int, diversity_quantiles: np.ndarray) -> str:
            for i, q in enumerate(diversity_quantiles):
                if diversity <= q:
                    return f'ngram_q{i+1}'
            return f'ngram_q{len(diversity_quantiles)+1}'

        def get_starting_n_gram(sentence: str, n: int = 5) -> str:
            words = get_words(sentence)
            return ' '.join(words[:n]) if len(words) >= n else sentence

        def categorize_based_on_quantiles(value: float, quantiles: np.ndarray) -> str:
            for i, q in enumerate(quantiles):
                if value <= q:
                    return f'q{i+1}'
            return f'q{len(quantiles)+1}'

        def determine_quantiles(values: List[float], num_quantiles: int) -> np.ndarray:
            quantile_values = np.linspace(0, 1, num_quantiles + 2)[1:-1]
            return np.quantile(values, quantile_values)
        sentence_counts = [len(item.split()) for item in data]
        ttrs = [calculate_ttr(item) for item in data]
        num_length_quantiles = min(max_q, min(
            5, len(set(sentence_counts)) // 20))
        num_ttr_quantiles = min(max_q, min(5, len(set(ttrs)) // 20))
        length_quantiles = determine_quantiles(
            sentence_counts, num_length_quantiles)
        ttr_quantiles = determine_quantiles(ttrs, num_ttr_quantiles)
        ngram_diversities = [calculate_n_gram_diversity(
            n_gram_frequency(item)) for item in data]
        num_ngram_quantiles = min(max_q, min(
            5, len(set(ngram_diversities)) // 20))
        ngram_quantiles = determine_quantiles(
            ngram_diversities, num_ngram_quantiles)
        starting_ngrams = [get_starting_n_gram(item) for item in data]
        starting_ngram_freq = Counter(starting_ngrams)
        starting_ngram_counts = list(starting_ngram_freq.values())
        num_starting_ngram_quantiles = min(max_q, min(
            5, len(set(starting_ngram_counts)) // 20))
        starting_ngram_quantiles = determine_quantiles(
            starting_ngram_counts, num_starting_ngram_quantiles)
        starting_ngram_categories = {}
        for ngram in starting_ngram_freq:
            ngram_count = starting_ngram_freq[ngram]
            starting_ngram_category = categorize_based_on_quantiles(
                ngram_count, starting_ngram_quantiles)
            starting_ngram_categories[ngram] = starting_ngram_category
        processed_data = []
        ttr_class_distribution = Counter()
        sentence_length_distribution = Counter()
        n_gram_diversity_distribution = Counter()
        starting_ngram_distribution = Counter()
        for item in data:
            source_sentence = item
            ttr = calculate_ttr(source_sentence)
            ttr_class = calculate_ttr_class(ttr, ttr_quantiles)
            sentence_length = categorize_sentence_length(
                source_sentence, length_quantiles)
            n_gram_diversity = calculate_n_gram_diversity(
                n_gram_frequency(source_sentence))
            n_gram_diversity_class = categorize_n_gram_diversity(
                n_gram_diversity, ngram_quantiles)
            starting_ngram = get_starting_n_gram(source_sentence)
            starting_ngram_category = starting_ngram_categories[starting_ngram]
            ttr_class_distribution[ttr_class] += 1
            sentence_length_distribution[sentence_length] += 1
            n_gram_diversity_distribution[n_gram_diversity_class] += 1
            starting_ngram_distribution[starting_ngram_category] += 1
            processed_item: ProcessedDataString = {
                'source': source_sentence,
                'category_values': [ttr_class, sentence_length, n_gram_diversity_class, starting_ngram_category]
            }
            processed_data.append(processed_item)
        print("TTR Class Distribution:", dict(ttr_class_distribution))
        print("Sentence Length Distribution:",
              dict(sentence_length_distribution))
        print("N-Gram Diversity Distribution:",
              dict(n_gram_diversity_distribution))
        print("Starting N-Gram Distribution:",
              dict(starting_ngram_distribution))
        return processed_data

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

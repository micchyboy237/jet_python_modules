from typing import List, TypedDict
import numpy as np
from collections import Counter
from jet.data.stratified_sampler_helpers.sampler_types import ProcessedDataString
from jet.wordnet.words import get_words
from jet.wordnet.n_grams import n_gram_frequency, calculate_n_gram_diversity


class DataLabeler:
    """Handles labeling of data based on linguistic features like TTR, sentence length, n-gram diversity, and starting n-grams."""

    def __init__(self, data: List[str], max_quantiles: int = 2):
        """Initialize with data and maximum number of quantiles."""
        if not data:
            raise ValueError("Input data cannot be empty")
        self.data = data
        self.max_quantiles = max_quantiles

    def calculate_ttr(self, sentence: str) -> int:
        """Calculate Type-Token Ratio (TTR) for a sentence."""
        words = sentence.split()
        unique_words = set(words)
        return len(unique_words)

    def calculate_ttr_class(self, ttr: int, ttr_quantiles: np.ndarray) -> str:
        """Categorize TTR into quantile-based classes."""
        for i, q in enumerate(ttr_quantiles):
            if ttr <= q:
                return f'ttr_q{i+1}'
        return f'ttr_q{len(ttr_quantiles)+1}'

    def categorize_sentence_length(self, sentence: str, length_quantiles: np.ndarray) -> str:
        """Categorize sentence length into quantile-based classes."""
        word_count = len(sentence.split())
        for i, q in enumerate(length_quantiles):
            if word_count <= q:
                return f'q{i+1}'
        return f'q{len(length_quantiles)+1}'

    def categorize_n_gram_diversity(self, diversity: int, diversity_quantiles: np.ndarray) -> str:
        """Categorize n-gram diversity into quantile-based classes."""
        for i, q in enumerate(diversity_quantiles):
            if diversity <= q:
                return f'ngram_q{i+1}'
        return f'ngram_q{len(diversity_quantiles)+1}'

    def get_starting_n_gram(self, sentence: str, n: int = 5) -> str:
        """Extract the starting n-gram from a sentence."""
        words = get_words(sentence)
        return ' '.join(words[:n]) if len(words) >= n else sentence

    def categorize_based_on_quantiles(self, value: float, quantiles: np.ndarray) -> str:
        """Categorize a value into quantile-based classes."""
        for i, q in enumerate(quantiles):
            if value <= q:
                return f'q{i+1}'
        return f'q{len(quantiles)+1}'

    def determine_quantiles(self, values: List[float], num_quantiles: int) -> np.ndarray:
        """Determine quantiles for a list of values."""
        quantile_values = np.linspace(0, 1, num_quantiles + 2)[1:-1]
        return np.quantile(values, quantile_values)

    def label_data(self) -> List[ProcessedDataString]:
        """Label data with categories based on linguistic features."""
        # Calculate metrics
        sentence_counts = [len(item.split()) for item in self.data]
        ttrs = [self.calculate_ttr(item) for item in self.data]
        ngram_diversities = [calculate_n_gram_diversity(
            n_gram_frequency(item)) for item in self.data]
        starting_ngrams = [self.get_starting_n_gram(
            item) for item in self.data]

        # Determine number of quantiles
        num_length_quantiles = min(self.max_quantiles, min(
            5, len(set(sentence_counts)) // 20))
        num_ttr_quantiles = min(
            self.max_quantiles, min(5, len(set(ttrs)) // 20))
        num_ngram_quantiles = min(self.max_quantiles, min(
            5, len(set(ngram_diversities)) // 20))
        starting_ngram_freq = Counter(starting_ngrams)
        starting_ngram_counts = list(starting_ngram_freq.values())
        num_starting_ngram_quantiles = min(self.max_quantiles, min(
            5, len(set(starting_ngram_counts)) // 20))

        # Calculate quantiles
        length_quantiles = self.determine_quantiles(
            sentence_counts, num_length_quantiles)
        ttr_quantiles = self.determine_quantiles(ttrs, num_ttr_quantiles)
        ngram_quantiles = self.determine_quantiles(
            ngram_diversities, num_ngram_quantiles)
        starting_ngram_quantiles = self.determine_quantiles(
            starting_ngram_counts, num_starting_ngram_quantiles)

        # Categorize starting n-grams
        starting_ngram_categories = {
            ngram: self.categorize_based_on_quantiles(
                starting_ngram_freq[ngram], starting_ngram_quantiles)
            for ngram in starting_ngram_freq
        }

        # Build processed data
        processed_data = []
        ttr_class_distribution = Counter()
        sentence_length_distribution = Counter()
        n_gram_diversity_distribution = Counter()
        starting_ngram_distribution = Counter()

        for item in self.data:
            source_sentence = item
            ttr = self.calculate_ttr(source_sentence)
            ttr_class = self.calculate_ttr_class(ttr, ttr_quantiles)
            sentence_length = self.categorize_sentence_length(
                source_sentence, length_quantiles)
            n_gram_diversity = calculate_n_gram_diversity(
                n_gram_frequency(source_sentence))
            n_gram_diversity_class = self.categorize_n_gram_diversity(
                n_gram_diversity, ngram_quantiles)
            starting_ngram = self.get_starting_n_gram(source_sentence)
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

        # Log distributions
        print("TTR Class Distribution:", dict(ttr_class_distribution))
        print("Sentence Length Distribution:",
              dict(sentence_length_distribution))
        print("N-Gram Diversity Distribution:",
              dict(n_gram_diversity_distribution))
        print("Starting N-Gram Distribution:",
              dict(starting_ngram_distribution))

        return processed_data

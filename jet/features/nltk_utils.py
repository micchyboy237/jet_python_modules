import nltk
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from typing import Union, List, Dict, Tuple, Optional

# Download required NLTK data (only needs to run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def get_word_counts_lemmatized(text: str) -> Dict[str, int]:
    """
    Get word count mappings from a text string with lemmatization, excluding stop words,
    sorted by count in descending order.

    Args:
        text (str): Input text string to analyze.

    Returns:
        Dict[str, int]: Dictionary with lemmatized words as keys and their counts as values, sorted by count descending.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())

    words = [
        token for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    counts = Counter(lemmatized_words)

    # Sort counts by descending frequency
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def get_word_sentence_combination_counts(text: Union[str, List[str]], n: Optional[int] = None, min_count: int = 1, in_sequence: bool = False) -> Union[Dict[Tuple[str, ...], int], List[Dict[Tuple[str, ...], int]]]:
    """
    Get counts of word combinations (n-grams) within sentences from a text string or list of strings with lemmatization,
    excluding stop words, sorted by count in descending order, including only those with counts >= min_count.
    Combinations can be sequential or order-independent within each sentence. If n is None, counts combinations of all sizes.

    Args:
        text (Union[str, List[str]]): Input text as a single string or a list of strings to analyze.
        n (Optional[int]): Size of word combinations (default is None, meaning all sizes from 1 to max words in a sentence).
        min_count (int): Minimum count threshold for combinations to be included (default is 1).
        in_sequence (bool): If True, combinations are sequential n-grams (words must appear in order).
                            If False, combinations are order-independent (default is False).

    Returns:
        Union[Dict[Tuple[str, ...], int], List[Dict[Tuple[str, ...], int]]]: If input is a string, returns a dictionary
            with tuples of lemmatized word combinations as keys and their counts as values, sorted by count descending.
            If input is a list of strings, returns a list of such dictionaries, one for each input string.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process_single_text(text: str) -> Dict[Tuple[str, ...], int]:
        # Split text into sentences
        sentences = sent_tokenize(text.lower())

        # Initialize counter for word combinations
        counts = Counter()

        for sentence in sentences:
            # Tokenize sentence
            tokens = word_tokenize(sentence)

            # Filter alphabetic tokens and remove stop words
            words = [
                token for token in tokens
                if token.isalpha() and token not in stop_words
            ]

            # Lemmatize words
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

            # Determine combination sizes
            if n is None:
                # Max combination size is number of words in sentence
                max_n = len(lemmatized_words)
                combination_sizes = range(1, max_n + 1)
            else:
                combination_sizes = [n]

            # Generate combinations for each size
            for current_n in combination_sizes:
                if in_sequence:
                    # Generate sequential n-grams
                    combinations = [
                        tuple(lemmatized_words[i:i + current_n])
                        for i in range(len(lemmatized_words) - current_n + 1)
                    ]
                else:
                    # Generate all possible n-word combinations within the sentence
                    combinations = list(itertools.combinations(
                        lemmatized_words, current_n))

                # Update counter with combinations
                counts.update(combinations)

        # Filter by min_count and sort by descending frequency
        return dict(sorted(
            {ngram: count for ngram, count in counts.items() if count >=
             min_count}.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    # Handle input type
    if isinstance(text, str):
        return process_single_text(text)
    elif isinstance(text, list):
        return [process_single_text(item) for item in text]
    else:
        raise ValueError("Input must be a string or a list of strings")

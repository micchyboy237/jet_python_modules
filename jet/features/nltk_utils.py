import math
import nltk
import itertools
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from typing import Literal, Union, List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from functools import partial

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('R'):
        return 'r'
    return 'n'


def process_single_text(single_text: str, n: Optional[int] = None, min_count: int = 1,
                        in_sequence: bool = False, max_n: int = 10) -> Dict[str, int]:
    if not single_text.strip():
        return {}
    counts = Counter()
    sentences = sent_tokenize(single_text.lower())
    sentence_words = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words = [token for token in tokens if token.isalpha()
                 and token not in STOP_WORDS]
        tagged_words = pos_tag(words)
        lemmatized_words = []
        for word, tag in tagged_words:
            pos = get_wordnet_pos(tag)
            # Force verb lemmatization for third-person singular forms
            if pos == 'v' and tag in ('VBZ', 'VBP'):
                lemmatized = LEMMATIZER.lemmatize(word, pos)
                # Additional check for common verb endings
                if word.endswith('s') and lemmatized == word:
                    lemmatized = LEMMATIZER.lemmatize(word[:-1], pos)
                elif word.endswith('es') and lemmatized == word:
                    lemmatized = LEMMATIZER.lemmatize(word[:-2], pos)
            else:
                lemmatized = LEMMATIZER.lemmatize(word, pos)
            lemmatized_words.append(lemmatized)
        if lemmatized_words:
            sentence_words.append(lemmatized_words)
    if not sentence_words:
        return {}
    if n is None:
        max_n_local = min(max(len(words) for words in sentence_words), max_n)
        combination_sizes = range(1, max_n_local + 1)
    else:
        combination_sizes = [min(n, max_n)]
    for lemmatized_words in sentence_words:
        for current_n in combination_sizes:
            if current_n > len(lemmatized_words):
                continue
            if in_sequence:
                combinations = [
                    ','.join(lemmatized_words[i:i + current_n])
                    for i in range(len(lemmatized_words) - current_n + 1)
                ]
            else:
                combinations = [','.join(c) for c in itertools.combinations(
                    lemmatized_words, current_n)]
            counts.update(combinations)
    result = dict(sorted(
        {ngram: count for ngram, count in counts.items() if count >=
         min_count}.items(),
        key=lambda x: x[1], reverse=True
    ))
    return result


def process_wrapper(text_item: str, n: Optional[int] = None, min_count: int = 1,
                    in_sequence: bool = False, max_n: int = 10) -> Dict[str, int]:
    return process_single_text(text_item, n, min_count, in_sequence, max_n)


def get_word_sentence_combination_counts(
    text: Union[str, List[str]],
    n: Optional[int] = None,
    min_count: int = 1,
    in_sequence: bool = False,
    max_n: int = 10,
    show_progress: bool = True
) -> Union[Dict[str, int], List[Dict[str, int]]]:
    if isinstance(text, str):
        return process_single_text(text, n, min_count, in_sequence, max_n)
    elif isinstance(text, list):
        # Process all texts to get combinations
        all_combinations_by_text = []
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        process_func = partial(
            process_wrapper,
            n=n,
            min_count=1,  # Set min_count=1 to collect all combinations initially
            in_sequence=in_sequence,
            max_n=max_n
        )

        if len(text) <= 10 or n == 1:
            all_combinations_by_text = [
                process_wrapper(t, n, 1, in_sequence, max_n) for t in text
            ]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                try:
                    all_combinations_by_text = list(tqdm(
                        executor.map(process_func, text),
                        total=len(text),
                        desc="Processing texts",
                        disable=not show_progress
                    ))
                except Exception as e:
                    print(f"Error in multiprocessing: {e}")
                    raise

        # Aggregate all combinations to determine valid ones based on total counts
        total_counts = Counter()
        for text_counts in all_combinations_by_text:
            total_counts.update(text_counts)

        # Identify combinations that meet the min_count threshold globally
        valid_combinations = {ngram for ngram,
                              count in total_counts.items() if count >= min_count}

        # Filter each text's counts to include only valid combinations
        result = []
        for text_counts in all_combinations_by_text:
            filtered_counts = {
                ngram: count for ngram, count in text_counts.items() if ngram in valid_combinations
            }
            sorted_counts = dict(
                sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))
            result.append(sorted_counts)

        return result
    else:
        raise ValueError("Input must be a string or a list of strings")


def process_single_text_simple(single_text: str) -> Dict[str, int]:
    if not single_text.strip():
        return {}
    tokens = word_tokenize(single_text.lower())
    words = [token for token in tokens if token.isalpha()
             and token not in STOP_WORDS]
    if not words:
        return {}
    tagged_words = pos_tag(words)
    lemmatized_words = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_words
    ]
    counts = Counter(lemmatized_words)
    result = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return result


def get_word_counts_lemmatized(
    text: Union[str, List[str]],
    pos: Optional[List[Literal['noun', 'verb', 'adjective', 'adverb']]] = None,
    min_count: int = 1,
    with_score: bool = False,
    percent_threshold: float = 0.0
) -> Union[Dict[str, Union[int, float]], List[Dict[str, Union[int, float]]]]:
    """
    Get word count mappings from a text string or list of strings with lemmatization, excluding stop words,
    sorted by count in descending order. Optionally filter by parts of speech, minimum count, and percentage threshold.
    For a list of strings, min_count and percent_threshold are applied to the total combined count across all strings.
    If with_score is True, returns scores normalized to percentages based on the maximum score.

    Args:
        text (Union[str, List[str]]): Input text string or list of strings to analyze.
        pos (Optional[List[Literal['noun', 'verb', 'adjective', 'adverb']]]): List of POS to include (e.g., ['noun', 'verb']). 
            If None, includes all words. Defaults to None.
        min_count (int): Minimum count threshold for words to be included. Defaults to 1.
        with_score (bool): If True, return scores normalized to percentages based on count and word length. Defaults to False.
        percent_threshold (float): Minimum percentage of total word count for words to be included (e.g., 5.0 for 5%). Defaults to 0.0.

    Returns:
        Union[Dict[str, Union[int, float]], List[Dict[str, Union[int, float]]]]: Dictionary with lemmatized words as keys 
            and their counts or percentage-normalized scores as values, or list of such dictionaries if input is a list, 
            sorted by count/score in descending order.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Map POS to Treebank tags
    pos_mapping = {
        'noun': 'N',
        'verb': 'V',
        'adjective': 'J',
        'adverb': 'R'
    }

    def process_single_text(single_text: str) -> List[tuple[str, str]]:
        # Tokenize and convert to lowercase
        tokens = word_tokenize(single_text.lower())

        # Filter alphabetic tokens and remove stop words
        words = [token for token in tokens if token.isalpha()
                 and token not in stop_words]

        # Get POS tags for words
        tagged_words = pos_tag(words)

        # Lemmatize and filter based on POS if provided
        lemmatized_words = []
        for word, tag in tagged_words:
            wordnet_pos = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)

            # If pos is provided, only include words matching the specified POS
            if pos is None or any(tag.startswith(pos_mapping[p]) for p in pos):
                lemmatized_words.append((lemmatized_word, tag))

        return lemmatized_words

    def calculate_scores(counts: Dict[str, int], total_words: int) -> Dict[str, float]:
        """Calculate scores based on word counts and word length, normalized to percentages."""
        if not counts:
            return {}
        # Compute raw scores
        raw_scores = {
            word: count * (1 + math.log(len(word)))
            for word, count in counts.items()
        }
        # Normalize to percentage of max score
        max_score = max(raw_scores.values(), default=1.0)
        return {
            word: (score / max_score) * 100
            for word, score in raw_scores.items()
        }

    if isinstance(text, str):
        lemmatized_words = [word for word, _ in process_single_text(text)]
        total_words = len(lemmatized_words)
        counts = Counter(lemmatized_words)
        min_count_threshold = max(min_count, int(
            total_words * percent_threshold / 100.0))
        filtered_counts = {word: count for word,
                           count in counts.items() if count >= min_count_threshold}

        if with_score and total_words > 0:
            result = calculate_scores(filtered_counts, total_words)
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        return dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))

    elif isinstance(text, list):
        # Process all texts and keep track of words and their tags
        all_words_by_text = [process_single_text(
            single_text) for single_text in text]

        # Get total counts across all texts for min_count and percent_threshold filtering
        all_words = [
            word for text_words in all_words_by_text for word, _ in text_words]
        total_words = len(all_words)
        total_counts = Counter(all_words)

        # Filter words based on min_count and percent_threshold
        min_count_threshold = max(min_count, int(
            total_words * percent_threshold / 100.0))
        valid_words = {word for word, count in total_counts.items()
                       if count >= min_count_threshold}

        # Create per-text counts or scores
        result = []
        for text_words in all_words_by_text:
            # Count only valid words for this text, respecting POS filter
            text_counts = Counter(
                word for word, tag in text_words if word in valid_words)
            total_words_per_text = len([word for word, _ in text_words])

            if with_score and total_words_per_text > 0:
                sorted_counts = calculate_scores(
                    text_counts, total_words_per_text)
            else:
                sorted_counts = text_counts

            result.append(
                dict(sorted(sorted_counts.items(), key=lambda x: x[1], reverse=True)))

        return result

    else:
        raise TypeError("Input must be a string or a list of strings")

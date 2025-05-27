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
        if len(text) <= 10 or n == 1:
            results = [
                process_wrapper(t, n, min_count, in_sequence, max_n) for t in text
            ]
            return results  # Include empty dictionaries
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            try:
                process_func = partial(
                    process_wrapper,
                    n=n,
                    min_count=min_count,
                    in_sequence=in_sequence,
                    max_n=max_n
                )
                results = list(tqdm(
                    executor.map(process_func, text),
                    total=len(text),
                    desc="Processing texts",
                    disable=not show_progress
                ))
                return results  # Include empty dictionaries
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                raise
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


def get_word_counts_lemmatized(text: Union[str, List[str]], pos: Optional[List[Literal['noun', 'verb', 'adjective', 'adverb']]] = None, min_count: int = 1) -> Union[Dict[str, int], List[Dict[str, int]]]:
    """
    Get word count mappings from a text string or list of strings with lemmatization, excluding stop words,
    sorted by count in descending order. Optionally filter by parts of speech and minimum count.
    For a list of strings, min_count is applied to the total combined count across all strings.

    Args:
        text (Union[str, List[str]]): Input text string or list of strings to analyze.
        pos (Optional[List[Literal['noun', 'verb', 'adjective', 'adverb']]]): List of POS to include (e.g., ['noun', 'verb']). 
            If None, includes all words. Defaults to None.
        min_count (int): Minimum count threshold for words to be included. Defaults to 1.

    Returns:
        Union[Dict[str, int], List[Dict[str, int]]]: Dictionary with lemmatized words as keys and their counts as values,
            or list of such dictionaries if input is a list, sorted by count in descending order.
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

    if isinstance(text, str):
        lemmatized_words = [word for word, _ in process_single_text(text)]
        counts = Counter(lemmatized_words)
        filtered_counts = {word: count for word,
                           count in counts.items() if count >= min_count}
        return dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))

    elif isinstance(text, list):
        # Process all texts and keep track of words and their tags
        all_words_by_text = [process_single_text(
            single_text) for single_text in text]

        # Get total counts across all texts for min_count filtering
        all_words = [
            word for text_words in all_words_by_text for word, _ in text_words]
        total_counts = Counter(all_words)

        # Filter words based on min_count
        valid_words = {word for word,
                       count in total_counts.items() if count >= min_count}

        # Create per-text counts, including only valid words
        result = []
        for text_words in all_words_by_text:
            # Count only valid words for this text, respecting POS filter
            text_counts = Counter(
                word for word, tag in text_words if word in valid_words)
            sorted_counts = dict(
                sorted(text_counts.items(), key=lambda x: x[1], reverse=True))
            result.append(sorted_counts)

        return result

    else:
        raise TypeError("Input must be a string or a list of strings")

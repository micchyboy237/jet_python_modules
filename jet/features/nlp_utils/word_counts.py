import math
from math import ceil
from collections import Counter
from typing import List, Dict, Union, Optional, Literal
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from .utils import get_wordnet_pos

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def get_word_counts_lemmatized(
    text: Union[str, List[str]],
    pos: Optional[List[Literal["noun", "verb", "adjective", "adverb"]]] = None,
    min_count: int = 1,
    as_score: bool = False,
    percent_threshold: float = 0.0
) -> Union[Dict[str, Union[int, float]], List[Dict[str, Union[int, float]]]]:
    """
    Get word count mappings from a text string or list of strings with lemmatization, excluding stop words,
    sorted by count in descending order. Optionally filter by parts of speech, minimum count, and percentage threshold.
    For a list of strings, min_count and percent_threshold are applied to the total combined count across all strings.
    If as_score is True, returns scores normalized to percentages based on the maximum score.

    Args:
        text (Union[str, List[str]]): Input text string or list of strings to analyze.
        pos (Optional[List[Literal['noun', 'verb', 'adjective', 'adverb']]]): List of POS to include (e.g., ['noun', 'verb']). 
            If None, includes all words. Defaults to None.
        min_count (int): Minimum count threshold for words to be included. Defaults to 1.
        as_score (bool): If True, return scores normalized to percentages based on count and word length. Defaults to False.
        percent_threshold (float): Minimum percentage of total word count for words to be included (e.g., 5.0 for 5%). Defaults to 0.0.

    Returns:
        Union[Dict[str, Union[int, float]], List[Dict[str, Union[int, float]]]]: Dictionary with lemmatized words as keys 
            and their counts or percentage-normalized scores as values, or list of such dictionaries if input is a list, 
            sorted by count/score in descending order.
    """

    pos_mapping = {"noun": "N", "verb": "V", "adjective": "J", "adverb": "R"}

    def process_single_text(single_text: str) -> List[tuple[str, str]]:
        tokens = word_tokenize(single_text.lower())
        words = [t for t in tokens if (
            t.isalpha() or t.isdigit()) and t not in stop_words]
        tagged_words = pos_tag(words)

        result = []
        for word, tag in tagged_words:
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            if pos is None or any(tag.startswith(pos_mapping[p]) for p in pos):
                result.append((lemma, tag))
        return result

    def calculate_scores(counts: Dict[str, int], total_words: int) -> Dict[str, float]:
        raw_scores = {w: c * (1 + math.log(len(w))) for w, c in counts.items()}
        max_score = max(raw_scores.values(), default=1.0)
        return {w: (s / max_score) * 100 for w, s in raw_scores.items()}

    if isinstance(text, str):
        lemmatized_words = [w for w, _ in process_single_text(text)]
        total_words = len(lemmatized_words)
        counts = Counter(lemmatized_words)
        min_threshold = max(min_count, ceil(
            total_words * percent_threshold / 100.0))
        filtered = {w: c for w, c in counts.items() if c >= min_threshold}

        return (
            dict(sorted(calculate_scores(filtered, total_words).items(),
                 key=lambda x: x[1], reverse=True))
            if as_score and total_words > 0
            else dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))
        )

    elif isinstance(text, list):
        all_words_by_text = [process_single_text(t) for t in text]
        all_words = [w for words in all_words_by_text for w, _ in words]
        total_counts = Counter(all_words)
        total_words = len(all_words)
        min_threshold = max(min_count, ceil(
            total_words * percent_threshold / 100.0))
        valid_words = {w for w, c in total_counts.items() if c >=
                       min_threshold}

        results = []
        for words in all_words_by_text:
            counts = Counter(w for w, _ in words if w in valid_words)
            total_in_doc = sum(counts.values())
            if as_score and total_in_doc > 0:
                counts = calculate_scores(counts, total_in_doc)
            results.append(
                dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)))
        return results

    else:
        raise TypeError("Input must be a string or a list of strings")

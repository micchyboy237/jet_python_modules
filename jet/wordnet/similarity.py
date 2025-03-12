import re
import json

from typing import List, Optional, TypedDict, Union
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.llm.utils.embeddings import get_ollama_embedding_function
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from jet.wordnet.words import get_words
from jet.logger import logger, time_it
from difflib import SequenceMatcher, ndiff, get_close_matches, unified_diff
from tqdm import tqdm
# from instruction_generator.wordnet.SpellingCorrectorNorvig import SpellingCorrectorNorvig
from jet.wordnet.wordnet_types import FilterResult, SimilarityResult

DEFAULT_SENTENCE_EMBED_MODEL = "paraphrase-MiniLM-L12-v2"


def sentence_similarity(base_sentence: str, sentences_to_compare: Union[str, List[str]], *, model_name: str | OLLAMA_EMBED_MODELS = DEFAULT_SENTENCE_EMBED_MODEL) -> List[float]:
    # Convert a single string to a list
    if isinstance(sentences_to_compare, str):
        sentences_to_compare = [sentences_to_compare]

    if model_name not in OLLAMA_EMBED_MODELS.__args__:
        # model = SentenceTransformer('Ramos-Ramos/xlm-roberta-base-en-tl-4-end')
        # model = SentenceTransformer("danjohnvelasco/filipino-sentence-roberta-v1")
        # model = SentenceTransformer('meedan/paraphrase-filipino-mpnet-base-v2')
        model = SentenceTransformer(model_name)
        base_embedding = model.encode([base_sentence])[0]
        embeddings = model.encode(sentences_to_compare)
    else:
        model_name: OLLAMA_EMBED_MODELS = model_name
        embed_func = get_ollama_embedding_function(model_name)
        base_embedding = embed_func(base_sentence)
        embeddings = embed_func(sentences_to_compare)

    return [1 - cosine(base_embedding, emb) for emb in embeddings]


@time_it
def filter_highest_similarity(query: str, candidates: List[str], *, model_name: str = DEFAULT_SENTENCE_EMBED_MODEL, threshold: Optional[float] = None) -> FilterResult:
    if not candidates:
        raise ValueError("No candidates provided for comparison.")

    similarities = sentence_similarity(
        query, candidates, model_name=model_name)
    highest_similarity_score = max(similarities)
    highest_similarity_text = candidates[similarities.index(
        highest_similarity_score)]

    others = [
        {
            'text': candidates[i],
            'score': similarities[i],
            'percent_difference': 100 * (highest_similarity_score - similarities[i]) / highest_similarity_score
        }
        for i in range(len(candidates))
        if candidates[i] != highest_similarity_text
        and (not threshold or similarities[i] >= threshold)
    ]
    others.sort(key=lambda x: x['score'], reverse=True)

    return {
        'text': highest_similarity_text,
        'score': highest_similarity_score,
        'others': others
    }


@time_it
def search_similarities(query: str, candidates: List[str], *, model_name: str = DEFAULT_SENTENCE_EMBED_MODEL, threshold: Optional[float] = None) -> List[SimilarityResult]:
    if not candidates:
        raise ValueError("No candidates provided for comparison.")

    similarities = sentence_similarity(
        query, candidates, model_name=model_name)
    highest_similarity_score = max(similarities)
    highest_similarity_text = candidates[similarities.index(
        highest_similarity_score)]

    results = [
        {
            'text': candidates[i],
            'score': similarities[i],
            'percent_difference': 100 * (highest_similarity_score - similarities[i]) / highest_similarity_score
        }
        for i in range(len(candidates))
        if candidates[i] != highest_similarity_text
        and (not threshold or similarities[i] >= threshold)
    ]
    results.sort(key=lambda x: x['score'], reverse=True)

    return results


def is_not_alnum(s):
    return not s.isalnum()


def score_texts_similarity(text1, text2, isjunk=is_not_alnum):
    # Create a SequenceMatcher with isjunk function to ignore non-alphanumeric characters
    score = SequenceMatcher(isjunk, text1, text2, autojunk=False).ratio()
    return score


def are_texts_similar(text1, text2, threshold=0.7):
    is_similar = score_texts_similarity(text1, text2) >= threshold
    return is_similar


def filter_similar_texts(texts: List[str], threshold: float = 0.7) -> List[str]:
    filtered_texts = []
    for text in texts:
        # Add text to filtered_texts if it is similar to at least one text already in filtered_texts
        if any(are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            continue  # Skip adding the text if it's similar to any text in filtered_texts
        filtered_texts.append(text)
    return filtered_texts


def filter_different_texts(texts, threshold=0.7):
    filtered_texts = []
    for text in texts:
        if all(not are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            filtered_texts.append(text)
    return filtered_texts


def get_similar_texts(texts: List[str], threshold: float = 0.7) -> List[dict[str, str]]:
    """Return a list of dictionaries with similar text pairs and their similarity score based on the given threshold."""
    similar_text_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = score_texts_similarity(texts[i], texts[j])
            if similarity_score >= threshold:
                similar_text_pairs.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'score': similarity_score
                })
    return similar_text_pairs


def get_different_texts(texts: List[str], threshold: float = 0.7) -> List[dict[str, str]]:
    """Return a list of dictionaries with different text pairs and their similarity score based on the given threshold."""
    different_text_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity_score = score_texts_similarity(texts[i], texts[j])
            if similarity_score < threshold:
                different_text_pairs.append({
                    'text1': texts[i],
                    'text2': texts[j],
                    'score': similarity_score
                })
    return different_text_pairs


def differences(texts: List[str], **kwargs) -> List[dict[str, str]]:
    all_differences = []
    for i in range(len(texts) - 1):
        diff = ndiff(texts[i].split(), texts[i + 1].split(), **kwargs)
        differences = [line[2:] for line in diff if line.startswith(
            '+ ') or line.startswith('- ')]
        all_differences.append(
            {'text1': texts[i], 'text2': texts[i + 1], 'differences': differences})
    return all_differences


def similars(texts: List[str], **kwargs) -> List[dict[str, str]]:
    all_similars = []
    for i in range(len(texts) - 1):
        diff = ndiff(texts[i].split(), texts[i + 1].split(), **kwargs)
        similars = [line.strip() for line in diff if not line.startswith(
            '+ ') and not line.startswith('- ')]
        all_similars.append(
            {'text1': texts[i], 'text2': texts[i + 1], 'similars': similars})
    return all_similars


def compare_text_pairs(texts: List[str], **kwargs) -> List[dict[str, List[str]]]:
    comparisons = []
    for i in range(len(texts) - 1):
        diff = list(ndiff(texts[i].split(), texts[i + 1].split(), **kwargs))
        similarities = [line.strip() for line in diff if line.startswith('  ')]
        differences = [line[2:] for line in diff if line.startswith(
            '+ ') or line.startswith('- ')]
        comparisons.append({
            'text1': texts[i],
            'text2': texts[i + 1],
            'similarities': similarities,
            'differences': differences
        })
    return comparisons


def has_close_match(text, texts, threshold=0.7) -> bool:
    # Use score_texts_similarity to check if the text has a close match in the list of texts
    for existing_text in texts:
        similarity_score = score_texts_similarity(text, existing_text)
        if similarity_score >= threshold:
            return True
    return False


def get_word_index_from_ngrams(word: str, ngrams: List[str]) -> int:
    # Find item in ngrams that contains the word, return -1 if not found
    return next((i for i, w in enumerate(ngrams) if word in w), -1)


def get_ngrams_by_word(word: str, text: str, n: int = 1, ignore_punctuation: bool = False) -> List[str]:
    word = word.lower()
    text = text.lower()

    words = get_words(text, n, ignore_punctuation=ignore_punctuation)

    # Filter ngrams that contain the word
    ngrams_with_word = [ngram for ngram in words if word in ngram]

    try:
        return ngrams_with_word[0]
    except IndexError as e:
        logger.error(f"{n}-word '{word}' not found in text '{text}'")
        raise e


def score_word_placement_similarity(word: str, text1: str, text2: str, n: int = 1) -> float:
    """
    Scores the similarity of the placement of a word in two texts, case-insensitively.
    The score is adjusted based on the positions of the word in both texts relative to the length of the longer text.
    """
    word = word.lower()
    text1 = text1.lower()
    text2 = text2.lower()

    word = get_ngrams_by_word(word, text1, n, ignore_punctuation=True)
    words1 = get_words(text1, n, ignore_punctuation=True)
    words2 = get_words(text2, n, ignore_punctuation=True)

    position1 = get_word_index_from_ngrams(word, words1)
    position2 = get_word_index_from_ngrams(word, words2)

    if position1 == -1 or position2 == -1:
        return 0.0

    max_length = max(len(words1), len(words2))
    # Normalize the position difference by the length of the longer text
    score = 1.0 - (abs(position1 - position2) / max_length)
    return score


def has_approximately_same_word_placement(word: str, text: str, texts: List[str], n: int = 1, threshold=0.8) -> bool:
    """
    Checks if the word has the approximately same relative position in the given text as in the list of texts,
    with a dynamically calculated threshold based on text lengths and word positions.
    """
    for existing_text in texts:
        try:
            similarity_score = score_word_placement_similarity(
                word, text, existing_text, n)
        except IndexError:
            continue
        if similarity_score >= threshold:
            return True
    return False


class TextComparator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.spell_corrector = SpellingCorrectorNorvig()

    @staticmethod
    def normalize(text):
        """Normalize texts by removing non-alphanumeric characters and converting to lower case."""
        result = re.sub(r'\W+', '', text).lower()
        return result

    def contains_segments(self, text1, text2):
        long_text = text1 if len(text1) > len(text2) else text2
        short_text = text2 if len(text1) > len(text2) else text1

        # Check if any of the texts are empty
        if not long_text or not short_text:
            return False

        # Split texts into lines and normalize each line
        normalized_long_lines = [self.normalize(
            line) for line in long_text.split('\n') if line.strip()]
        normalized_short_lines = [self.normalize(
            line) for line in short_text.split('\n') if line.strip()]

        # Ensure the list with fewer lines is considered the "shorter" one for comparison
        if len(normalized_long_lines) < len(normalized_short_lines):
            normalized_long_lines, normalized_short_lines = normalized_short_lines, normalized_long_lines

        # Check each segment from the shorter text against all segments in the longer text
        for short_line in normalized_short_lines:
            if not any(self.calculate_similarity_ratio(short_line, long_line) >= self.threshold for long_line in normalized_long_lines):
                return False
        return True

    def has_improved_spelling(self, updated_text, base_text):
        base_text_misspelled_words = self.spell_corrector.unknown_words(
            base_text)
        updated_text_misspelled_words = self.spell_corrector.unknown_words(
            updated_text)

        has_improved_spelling = updated_text_misspelled_words == 0 or len(updated_text_misspelled_words) < len(
            base_text_misspelled_words)
        return has_improved_spelling

    @staticmethod
    def calculate_similarity_ratio(text1, text2):
        """Calculate the similarity ratio based on the length of the longest common substring."""
        m = [[0] * (1 + len(text2)) for i in range(1 + len(text1))]
        longest = 0
        for x in range(1, 1 + len(text1)):
            for y in range(1, 1 + len(text2)):
                if text1[x - 1] == text2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    longest = max(longest, m[x][y])
                else:
                    m[x][y] = 0
        return longest / min(len(text1), len(text2))


if __name__ == '__main__':
    base_sentence = "October seven is the date of our vacation to Camarines Sur."
    sentences_to_compare = [
        "October 7 is our holiday in Camarines Sur.",
        "October 7 is the day we went on vacation to Camarines Sur.",
        "The seventh of October is the day of our vacation in Camarines Sur."
    ]

    {
        "text": "The seventh of October is the day of our vacation in Camarines Sur.",
        "score": 0.9571385864934139,
        "others": [
            {
                "text": "October 7 is the day we went on vacation to Camarines Sur.",
                "score": 0.9564081690435453,
                "percent_difference": 0.07631261137893704
            },
            {
                "text": "October 7 is our holiday in Camarines Sur.",
                "score": 0.898377777869796,
                "percent_difference": 6.139216353077435
            }
        ]
    }

    print(f"Base sentence:\n{base_sentence}")
    result = filter_highest_similarity(base_sentence, sentences_to_compare)
    print("Highest similarity result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

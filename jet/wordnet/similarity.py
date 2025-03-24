from typing import List, Optional, Literal
from sentence_transformers import util
import torch
from typing import Callable, List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from jet.llm.utils.embeddings import SFEmbeddingFunction, get_embedding_function
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Callable, Optional, TypedDict
from sklearn.cluster import KMeans
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

    embed_func = get_embedding_function(model_name)
    base_embedding = embed_func(base_sentence)
    embeddings = embed_func(sentences_to_compare)

    return [1 - cosine(base_embedding, emb) for emb in embeddings]


def filter_highest_similarity_old(query: str, candidates: List[str], *, model_name: str = DEFAULT_SENTENCE_EMBED_MODEL, threshold: Optional[float] = None) -> FilterResult:
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


def filter_highest_similarity(
    query: str,
    candidates: List[str],
    *,
    model_name: str = "paraphrase-MiniLM-L12-v2",
    similarity_metric: Literal["cosine", "dot", "euclidean"] = "cosine",
    threshold: Optional[float] = None
) -> FilterResult:
    if not candidates:
        raise ValueError("No candidates provided for comparison.")

    embed_func = get_embedding_function(model_name)

    # Compute embeddings (keeping original list format)
    query_embedding: list[float] = embed_func(query)  # 1D list
    candidate_embeddings: list[list[float]] = embed_func(candidates)  # 2D list

    # Convert to tensors for calculations
    query_tensor = torch.tensor(query_embedding)  # (D,)
    candidate_tensor = torch.tensor(candidate_embeddings)  # (N, D)

    # Compute similarity scores based on chosen metric
    if similarity_metric == "cosine":
        similarities = util.pytorch_cos_sim(
            query_tensor, candidate_tensor)[0].tolist()
    elif similarity_metric == "dot":
        similarities = torch.matmul(query_tensor, candidate_tensor.T).tolist()
    elif similarity_metric == "euclidean":
        similarities = (-torch.cdist(query_tensor.unsqueeze(0),
                        candidate_tensor.unsqueeze(0), p=2)[0]).tolist()
        # Negative because lower Euclidean distance means higher similarity
    else:
        raise ValueError(
            "Invalid similarity metric. Choose 'cosine', 'dot', or 'euclidean'.")

    # Find the best match
    sorted_results = sorted(zip(candidates, similarities),
                            key=lambda x: x[1], reverse=True)
    best_text, best_score = sorted_results[0]

    # Compute percent differences
    others = [
        {
            "text": text,
            "score": score,
            "percent_difference": (best_score - score) / best_score * 100 if best_score != 0 else 0
        }
        for text, score in sorted_results[1:]
    ]

    # Apply threshold filtering
    if threshold is not None:
        others = [item for item in others if item["score"] >= threshold]

    return {
        "text": best_text,
        "score": best_score,
        "others": others
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


def plot_text_embeddings(texts: List[str], embeddings: List[List[float]], title: str = "Text Embeddings Visualization"):
    """
    Plots text embeddings in a 2D space using PCA and auto-opens the viewer.

    Args:
        texts (List[str]): List of text inputs.
        embeddings (List[List[float]]): Corresponding embeddings.
        title (str): Title of the plot.
    """
    if len(embeddings) == 0 or len(embeddings[0]) < 2:
        raise ValueError(
            "Embeddings must have at least two dimensions for visualization.")

    # Reduce embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 7))
    plt.scatter(reduced_embeddings[:, 0],
                reduced_embeddings[:, 1], marker='o', alpha=0.7)

    # Annotate points with corresponding text (truncate long texts for readability)
    for i, text in enumerate(texts):
        plt.annotate(text[:20] + '...' if len(text) > 20 else text,
                     (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     fontsize=9, alpha=0.75)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.grid(True)

    # Automatically open the viewer
    plt.show()


def cluster_texts(
    texts: List[str],
    embedding_function: Callable[[List[str]], List[List[float]]],
    num_clusters: Optional[int] = None
) -> Dict[int, List[str]]:
    """
    Groups similar texts into clusters based on embeddings.

    Args:
        texts (List[str]): List of text inputs.
        embedding_function (Callable): Function to generate text embeddings.
        num_clusters (int, optional): Number of clusters. If None, it will be auto-determined.

    Returns:
        Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of similar texts.
    """

    # Generate embeddings
    embeddings = np.array(embedding_function(texts))

    # Auto-determine number of clusters if not provided
    if num_clusters is None:
        num_clusters = max(2, min(len(texts) // 3, 10))  # Dynamic selection

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Organize texts into clusters
    clustered_texts = {}
    for i, label in enumerate(cluster_labels):
        clustered_texts.setdefault(label, []).append(texts[i])

    return clustered_texts


class SimilarResult(TypedDict):
    text: str
    score: float


def find_most_similar_texts(
    texts: List[str],
    embedding_function: Callable[[List[str]], List[List[float]]],
    *,
    threshold: float = 0.25,
    num_decimal: int = 2
) -> Dict[str, List[SimilarResult]]:
    """
    Finds the most similar texts using cosine similarity.

    Args:
        texts (List[str]): List of text inputs.
        embedding_function (Callable): Function to generate text embeddings.
        threshold (float): Similarity threshold for grouping.
        num_decimal (int): Number of decimal places to truncate similarity scores.

    Returns:
        Dict[str, List[SimilarResult]]: Dictionary mapping each text to similar ones with scores.
    """
    embeddings = np.array(embedding_function(texts))
    similarity_matrix = cosine_similarity(embeddings)

    factor = 10 ** num_decimal  # Scaling factor for truncation

    text_with_scores = {}
    for i, text in enumerate(texts):
        score_results = [
            {"text": texts[j], "score": similarity_matrix[i, j]}
            for j in range(len(texts))
            # Truncate without rounding
            if (int(similarity_matrix[i, j] * factor) / factor) > threshold and i != j
        ]

        text_with_scores[text] = sorted(
            score_results, key=lambda x: x["score"], reverse=True
        )

    return text_with_scores


if __name__ == "__main__":
    # Sample texts with varying similarities and differences
    texts = [
        # Group 1: Technology
        "Artificial Intelligence is transforming industries.",
        "Machine Learning models predict outcomes using data.",
        "Deep Learning is a subset of machine learning.",
        "Neural networks simulate the human brain.",

        # Group 2: Space and Astronomy
        "NASA discovered a new exoplanet in the habitable zone.",
        "Black holes warp space-time due to their gravity.",
        "The James Webb Telescope captures deep-space images.",
        "Astrobiology explores the potential for extraterrestrial life.",

        # Group 3: Sports
        "Soccer is the world's most popular sport.",
        "Basketball requires agility and teamwork.",
        "Tennis matches can last for hours in Grand Slams.",
        "Formula 1 cars are designed for maximum speed and aerodynamics.",

        # Group 4: Nature & Environment
        "Climate change is affecting global weather patterns.",
        "Deforestation leads to habitat loss and species extinction.",
        "Renewable energy sources include solar and wind power.",
        "Oceans absorb a large percentage of the Earth's heat.",

        # Group 5: Random (for diversity)
        "Cooking is an art that blends flavors and techniques.",
        "Music has the power to evoke emotions and memories.",
        "Philosophy questions the nature of existence and reality.",
        "History teaches us lessons from past civilizations."
    ]

    # Generate embeddings (Replace with actual embedding function)
    embedding_function = SFEmbeddingFunction("paraphrase-MiniLM-L12-v2")
    embeddings = embedding_function(texts)

    # Plot the embeddings
    plot_text_embeddings(texts, embeddings)

    print("Done")

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


__all__ = [
    "sentence_similarity",
    "filter_highest_similarity",
    "search_similarities",
    "is_not_alnum",
    "score_texts_similarity",
    "are_texts_similar",
    "filter_similar_texts",
    "filter_different_texts",
    "get_similar_texts",
    "get_different_texts",
    "differences",
    "similars",
    "compare_text_pairs",
    "has_close_match",
    "get_word_index_from_ngrams",
    "get_ngrams_by_word",
    "score_word_placement_similarity",
    "has_approximately_same_word_placement",
    "TextComparator",
    "plot_text_embeddings",
    "cluster_texts",
    "SimilarResult",
    "find_most_similar_texts",
]

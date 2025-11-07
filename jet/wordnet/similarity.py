import json
import re
from collections import defaultdict
from difflib import SequenceMatcher, ndiff
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
from jet.data.utils import generate_key
from jet.logger import logger, time_it
from jet.models.embeddings.base import (
    get_embedding_function,
)
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.models.model_types import EmbedModelType
from jet.vectors.clusters.cluster_types import ClusteringMode
from jet.vectors.document_types import HeaderDocument, HeaderDocumentWithScore
from jet.wordnet.words import get_words
from jet.wordnet.wordnet_types import FilterResult, SimilarityResult
from scipy.spatial.distance import cosine
from sentence_transformers import util
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# from instruction_generator.wordnet.SpellingCorrectorNorvig import SpellingCorrectorNorvig

DEFAULT_EMBED_MODEL: EmbedModelType = "all-MiniLM-L12-v2"


class ClusterResult(TypedDict):
    label: int
    texts: List[str]


def sentence_similarity(base_sentence: str, sentences_to_compare: Union[str, List[str]], *, model_name: EmbedModelType = DEFAULT_EMBED_MODEL) -> List[float]:
    # Convert a single string to a list
    if isinstance(sentences_to_compare, str):
        sentences_to_compare = [sentences_to_compare]

    embed_func = get_embedding_function(model_name)
    base_embedding: list[float] = embed_func(base_sentence)
    embeddings: list[list[float]] = embed_func(sentences_to_compare)

    return [1 - cosine(base_embedding, emb) for emb in embeddings]


def get_text_groups(
    texts: List[str],
    threshold: float = 0.75,
    model_name: EmbedModelType = "all-MiniLM-L12-v2"
) -> List[List[str]]:
    """
    Groups similar texts into exclusive clusters based on cosine similarity.

    Args:
        texts (list[str]): A list of texts to cluster.
        threshold (float): Minimum similarity score required to be grouped. Default is 0.5.
        model_name (str): The embedding model name.

    Returns:
        List[List[str]]: A list of lists, where each inner list contains similar texts.
    """
    if not texts:
        raise ValueError("'texts' must be non-empty.")

    # Get embedding function
    embed_func = get_embedding_function(model_name)

    # Compute embeddings for all texts
    text_embeddings = np.array(embed_func(texts))

    # Normalize embeddings to speed up cosine similarity computation
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(text_embeddings, text_embeddings.T)

    # Track assigned texts
    assigned = set()
    groups = []

    for i, text in enumerate(texts):
        if text in assigned:
            continue  # Skip already assigned texts

        # Find texts most similar to the current text
        similarities = similarity_matrix[i]
        similar_indices = np.where(similarities >= threshold)[0]

        # Create a group and mark texts as assigned
        group = []
        for idx in similar_indices:
            if texts[idx] not in assigned:
                group.append(texts[idx])
                assigned.add(texts[idx])

        groups.append(group)

    return groups


def query_similarity_scores(
    query: Union[str, List[str]],
    texts: Union[str, List[str]],
    threshold: float = 0.0,
    model: Union[EmbedModelType, List[EmbedModelType]] = "all-MiniLM-L6-v2",
    fuse_method: Literal["average", "max", "min"] = "average",
    ids: Union[List[str], None] = None,
    metrics: Literal["cosine", "dot", "euclidean"] = "cosine"
) -> List[SimilarityResult]:
    """
    Computes similarity scores for queries against texts using one or more embedding models,
    fusing results into a single sorted list with one result per text.

    For each text and query, scores are averaged across models. Then, for each text,
    the query-specific scores are fused using the specified method ('average', 'max', or 'min').

    Args:
        query: Single query or list of queries.
        texts: Single text or list of texts to compare against.
        threshold: Minimum similarity score to include in results (default: 0.0).
        model: One or more embedding model names (default: "all-MiniLM-L6-v2").
        fuse_method: Fusion method for combining scores ('average', 'max', or 'min') (default: "average").
        ids: Optional list of IDs for texts; must match texts length if provided.
        metrics: Similarity metric to use ('cosine', 'euclidean', 'dot') (default: "cosine").

    Returns:
        List of SimilarityResult, containing one fused result per text,
        sorted by score in descending order with ranks, percent_difference, and doc_index.

    Raises:
        ValueError: If inputs are empty, model is empty, ids length mismatches texts,
                    invalid fuse_method, or invalid metrics.
    """
    if isinstance(query, str):
        query = [query]
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(model, str):
        model = [model]

    if not query or not texts:
        raise ValueError("Both query and texts must be non-empty.")
    if not model:
        raise ValueError("At least one model name must be provided.")
    if ids is not None and len(ids) != len(texts):
        raise ValueError(
            f"Length of ids ({len(ids)}) must match length of texts ({len(texts)})."
        )

    supported_methods = {"average", "max", "min"}
    if fuse_method not in supported_methods:
        raise ValueError(
            f"Fusion method must be one of {supported_methods}; got {fuse_method}."
        )

    supported_metrics = {"cosine", "euclidean", "dot"}
    if metrics not in supported_metrics:
        raise ValueError(
            f"Metrics must be one of {supported_metrics}; got {metrics}."
        )

    text_ids = (
        ids
        if ids is not None
        else [generate_key(text, query[0] if query else None) for text in texts]
    )

    # Collect all results across queries and models
    all_results: List[Dict[str, Any]] = []

    for model_name in model:
        embed_func = get_embedding_function(model_name)

        query_embeddings = np.array(embed_func(query))
        text_embeddings = np.array(embed_func(texts))

        if metrics == "cosine":
            # Normalize embeddings for cosine similarity
            query_norms = np.linalg.norm(
                query_embeddings, axis=1, keepdims=True)
            text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

            query_embeddings = np.divide(
                query_embeddings,
                query_norms,
                out=np.zeros_like(query_embeddings),
                where=query_norms != 0
            )
            text_embeddings = np.divide(
                text_embeddings,
                text_norms,
                out=np.zeros_like(text_embeddings),
                where=text_norms != 0
            )

            similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
        elif metrics == "dot":
            # Raw dot product without normalization
            similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
        elif metrics == "euclidean":
            # Euclidean distance (lower is better, so we negate and add 1 to make higher better)
            similarity_matrix = np.zeros((len(query), len(texts)))
            for i in range(len(query)):
                for j in range(len(texts)):
                    dist = np.linalg.norm(
                        query_embeddings[i] - text_embeddings[j])
                    similarity_matrix[i, j] = 1 / (1 + dist)

        for i, query_text in enumerate(query):
            scores = similarity_matrix[i]

            mask = scores >= threshold
            filtered_texts = np.array(texts)[mask]
            filtered_ids = np.array(text_ids)[mask]
            filtered_scores = scores[mask]
            filtered_indices = np.arange(
                len(texts))[mask]  # Track original indices

            sorted_indices = np.argsort(filtered_scores)[::-1]
            for idx, j in enumerate(sorted_indices):
                all_results.append({
                    "id": filtered_ids[j],
                    "doc_index": int(filtered_indices[j]),
                    "query": query_text,
                    "text": filtered_texts[j],
                    "score": float(filtered_scores[j]),
                })

    # Fuse results
    fused_results = fuse_all_results(all_results, method=fuse_method)

    # Update fused results to include doc_index
    # Fixed: Use result["id"] instead of result.id
    fused_dict = {result["id"]: result for result in fused_results}
    for result in all_results:
        if result["id"] in fused_dict:
            fused_dict[result["id"]]["doc_index"] = result["doc_index"]

    # Convert dictionaries to SimilarityResult TypedDict
    final_results = [
        {
            "id": result["id"],
            "rank": result["rank"],
            "doc_index": result.get("doc_index", 0),  # Default to 0 if not set
            "score": result["score"],
            "percent_difference": result["percent_difference"],
            "text": result["text"],
            "relevance": None,  # Optional field, not computed here
            "word_count": None  # Optional field, not computed here
        }
        for result in fused_dict.values()
    ]

    return final_results


def fuse_all_results(
    results: List[Dict[str, Any]],
    method: str = "average"
) -> List[SimilarityResult]:
    """
    Fuses similarity results into a single sorted list with one result per text.

    First, averages scores for each text and query across models.
    Then, fuses the query-specific scores for each text using the specified method.

    Args:
        results: List of result dictionaries with id, query, text, and score.
        method: Fusion method ('average', 'max', or 'min').

    Returns:
        List of SimilarityResult, sorted by score with ranks and percent_difference.

    Raises:
        ValueError: If an unsupported fusion method is provided.
    """
    # Step 1: Average scores for each (id, query, text) across models
    query_text_data = defaultdict(lambda: {"scores": [], "text": None})

    for result in results:
        key = (result["id"], result["query"], result["text"])
        query_text_data[key]["scores"].append(result["score"])
        query_text_data[key]["text"] = result["text"]

    query_text_averages = {
        key: {
            "text": data["text"],
            "score": float(sum(data["scores"]) / len(data["scores"]))
        }
        for key, data in query_text_data.items()
    }

    # Step 2: Collect query-specific scores for each (id, text)
    text_data = defaultdict(lambda: {"scores": [], "text": None})

    for (id_, query, text), data in query_text_averages.items():
        text_key = (id_, text)
        text_data[text_key]["scores"].append(data["score"])
        text_data[text_key]["text"] = text

    # Create fused results
    fused_scores = []
    if method == "average":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(sum(data["scores"]) / len(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    elif method == "max":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(max(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    elif method == "min":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(min(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    # Sort by score and assign ranks
    sorted_scores = sorted(
        fused_scores, key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(sorted_scores):
        result["rank"] = idx + 1

    # Calculate percent_difference
    if sorted_scores:
        max_score = sorted_scores[0]["score"]
        if max_score != 0:
            for result in sorted_scores:
                result["percent_difference"] = round(
                    abs(max_score - result["score"]) / max_score * 100, 2
                )
        else:
            for result in sorted_scores:
                result["percent_difference"] = 0.0

    return sorted_scores


def filter_highest_similarity_old(query: str, candidates: List[str], *, model_name: str = DEFAULT_EMBED_MODEL, threshold: Optional[float] = None) -> FilterResult:
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
    model_name: str = DEFAULT_EMBED_MODEL,
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
def search_similarities(query: str, candidates: List[str], *, model_name: str = DEFAULT_EMBED_MODEL, threshold: Optional[float] = None) -> List[SimilarityResult]:
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


def score_texts_similarity(text1: str, text2: str, isjunk: Callable[[str], bool] = None) -> float:
    matcher = SequenceMatcher(isjunk, text1.lower(),
                              text2.lower(), autojunk=False)
    score = matcher.ratio()
    # print(
    #     f"Similarity between '{text1[:30]}...' and '{text2[:30]}...': {score:.4f}")
    return score


def are_texts_similar(text1: str, text2: str, threshold: float = 0.7) -> bool:
    return score_texts_similarity(text1, text2) >= threshold


def filter_similar_texts(texts: List[str], threshold: float = 0.7, show_progress: bool = False) -> List[str]:
    filtered_texts = []
    iterator = tqdm(
        texts, desc="Filtering similar texts") if show_progress else texts
    for text in iterator:
        if any(are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            continue
        filtered_texts.append(text)
    print(f"Retained {len(filtered_texts)} similar texts")
    return filtered_texts


def filter_different_texts(texts: List[str], threshold: float = 0.7, show_progress: bool = False) -> List[str]:
    filtered_texts = []
    iterator = tqdm(
        texts, desc="Filtering different texts") if show_progress else texts
    for text in iterator:
        if all(not are_texts_similar(text, existing_text, threshold) for existing_text in filtered_texts):
            filtered_texts.append(text)
    print(f"Retained {len(filtered_texts)} diverse texts")
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


def preprocess_text(text: str) -> str:
    """
    Preprocesses a single text by normalizing whitespace, converting to lowercase,
    and removing special characters.

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # import re
    # # Convert to lowercase
    # text = text.lower()
    # # Remove special characters, keeping alphanumeric and spaces
    # text = re.sub(r'[^\w\s]', '', text)
    # # Normalize whitespace (replace multiple spaces with single space, strip)
    # text = re.sub(r'\s+', ' ', text).strip()
    return text


def group_similar_texts(
    texts: List[str],
    threshold: float = 0.8,
    model_name: str = "embeddinggemma",
    embeddings: Optional[np.ndarray] = None,
    ids: Optional[List[str]] = None,
    mode: ClusteringMode = "agglomerative"
) -> List[ClusterResult]:
    """
    Groups similar texts based on cosine similarity score using specified clustering mode, with deduplicated and preprocessed input texts.

    Args:
        texts (List[str]): List of input texts to be grouped.
        threshold (float): Similarity threshold for clustering. Default is 0.8.
        model_name (str): Sentence transformer model to use for embedding if embeddings not provided.
        embeddings (Optional[np.ndarray]): Precomputed embeddings as a NumPy array.
        ids (Optional[List[str]]): Optional list of IDs corresponding to texts. If provided, these will replace the text in the output.
        mode (ClusteringMode): Clustering method to use. Default is "agglomerative".

    Returns:
        List[ClusterResult]: List of dictionaries containing cluster labels and their corresponding original texts or IDs, including noise points (label -1).
    """
    if not texts:
        return []

    # Validate that ids, if provided, matches the length of texts
    if ids is not None and len(ids) != len(texts):
        raise ValueError("Length of ids must match length of texts")

    # Deduplicate texts while preserving order
    seen_texts = {}
    unique_texts = []
    original_texts = []
    original_ids = [] if ids is not None else None
    for i, text in enumerate(texts):
        if text not in seen_texts:
            seen_texts[text] = True
            # Preprocess text for embedding
            unique_texts.append(preprocess_text(text))
            original_texts.append(text)
            if ids is not None:
                original_ids.append(ids[i])

    # Load the embedding model if embeddings are not provided
    if embeddings is None:
        # model = SentenceTransformerRegistry.load_model(model_name)
        model = LlamacppEmbedding(model=model_name)
        embeddings = model.encode(unique_texts)

    # # Ensure embeddings is a 2D NumPy array (n_texts, embedding_dim)
    # embeddings_array = np.array(embeddings)
    # if embeddings_array.ndim != 2:
    #     raise ValueError(
    #         "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Compute cosine similarity matrix using NumPy
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / \
        np.maximum(norm, 1e-10)  # Avoid division by zero
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Ensure similarity matrix values are in [0, 1]
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    # Perform clustering based on mode
    if mode == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - threshold
        ).fit(1 - similarity_matrix)
    elif mode == "kmeans":
        n_texts = len(unique_texts)
        min_clusters = 1
        max_clusters = min(n_texts, max(2, n_texts // 2))
        best_n_clusters = min_clusters
        best_silhouette = -1
        best_labels = None

        for n_clusters in range(min_clusters, max_clusters + 1):
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            ).fit(normalized_embeddings)
            labels = clustering.labels_
            if n_clusters > 1:
                score = silhouette_score(
                    normalized_embeddings, labels, metric="euclidean")
                if score > best_silhouette:
                    best_silhouette = score
                    best_n_clusters = n_clusters
                    best_labels = labels
            else:
                best_labels = labels

        clustering.labels_ = best_labels
    elif mode == "dbscan":
        clustering = DBSCAN(
            eps=1 - threshold,
            min_samples=2,
            metric="precomputed"
        ).fit(1 - similarity_matrix)
    else:  # mode == "hdbscan"
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric="precomputed",
            cluster_selection_epsilon=1 - threshold
        ).fit((1 - similarity_matrix).astype(np.float64))

    # Organize texts or IDs into clusters, including noise points
    clusters: dict[int, List[str]] = {}
    for idx, label in enumerate(clustering.labels_):
        output_text = original_ids[idx] if ids is not None else original_texts[idx]
        clusters.setdefault(label, []).append(output_text)

    # Convert clusters to list of ClusterResult dictionaries
    return [{"label": label, "texts": texts} for label, texts in clusters.items()]


class GroupedResult(TypedDict):
    headers: List[str]
    contents: List[str]
    doc_indexes: List[int]
    source_urls: List[str]
    average_score: Optional[float]
    max_score: Optional[float]
    scores: Optional[List[float]]
    documents: Union[List[HeaderDocument], List[HeaderDocumentWithScore]]


def group_similar_headers(
    docs: Union[List[HeaderDocument], List[HeaderDocumentWithScore]],
    threshold: float = 0.7,
    model_name: EmbedModelType = "static-retrieval-mrl-en-v1"
) -> List[GroupedResult]:
    """
    Groups similar documents based on their header text similarity, preserving original documents.

    Args:
        docs (Union[List[HeaderDocument], List[HeaderDocumentWithScore]]): List of documents to group.
        threshold (float): Similarity threshold for clustering. Default is 0.7.
        model_name (EmbedModelType): Sentence transformer model to use for embedding.

    Returns:
        List[GroupedResult]: List of grouped results, each containing headers, contents, doc_indexes, source_urls, average_score, max_score, scores, and their original documents.
    """
    # Determine if docs is a list of HeaderDocumentWithScore
    is_with_score = (
        len(docs) > 0 and hasattr(
            docs[0], "score") and hasattr(docs[0], "node")
    )

    # Create text-document pairs
    try:
        if is_with_score:
            # For HeaderDocumentWithScore, use node.metadata
            text_doc_pairs: List[Tuple[str, HeaderDocumentWithScore]] = [
                (
                    doc.node.metadata["header"].lstrip('#').strip().lower(),
                    doc
                )
                for doc in docs
            ]
        else:
            text_doc_pairs: List[Tuple[str, HeaderDocument]] = [
                (
                    doc["metadata"]["header"].lstrip('#').strip().lower(),
                    doc
                )
                for doc in docs
            ]
        logger.info(
            f"group_similar_headers: Created {len(text_doc_pairs)} text-document pairs")
    except Exception as e:
        logger.error(
            f"group_similar_headers: Failed to create text-document pairs: {str(e)}")
        raise

    # Extract texts for grouping
    texts = [pair[0] for pair in text_doc_pairs]

    # Handle empty input
    if not texts:
        logger.error("group_similar_headers: Empty input list")
        return []

    # Group similar texts
    try:
        grouped_texts = group_similar_texts(
            texts, threshold=threshold, model_name=model_name)
        logger.info(
            f"group_similar_headers: Grouped into {len(grouped_texts)} clusters")
    except Exception as e:
        logger.error(f"group_similar_headers: Grouping failed: {str(e)}")
        raise

    # Map grouped texts back to original documents
    grouped_results: List[GroupedResult] = []
    try:
        for group in grouped_texts:
            group_docs = []
            group_headers = []
            group_doc_indexes = []
            group_source_urls = []
            group_contents = []
            seen_doc_indexes = set()
            seen_contents = set()
            group_scores = []
            for text in group:
                if is_with_score:
                    matching_docs = [pair[1]
                                     for pair in text_doc_pairs if pair[0] == text]
                    for doc in matching_docs:
                        doc_index = doc.node.metadata.get("doc_index", 0)
                        doc_content = doc.node.metadata.get("content", "")
                        if doc_index not in seen_doc_indexes and doc_content not in seen_contents:
                            seen_doc_indexes.add(doc_index)
                            seen_contents.add(doc_content)
                            group_docs.append(doc)
                            group_headers.append(
                                doc.node.metadata.get("header", ""))
                            group_doc_indexes.append(doc_index)
                            group_source_urls.append(
                                doc.node.metadata.get("source_url", ""))
                            group_contents.append(doc_content)
                            if hasattr(doc, "score") and doc.score is not None:
                                group_scores.append(doc.score)
                else:
                    matching_docs = [pair[1]
                                     for pair in text_doc_pairs if pair[0] == text]
                    for doc in matching_docs:
                        doc_index = doc["metadata"]["doc_index"]
                        doc_content = doc["metadata"]["content"]
                        if doc_index not in seen_doc_indexes and doc_content not in seen_contents:
                            seen_doc_indexes.add(doc_index)
                            seen_contents.add(doc_content)
                            group_docs.append(doc)
                            group_headers.append(doc["metadata"]["header"])
                            group_doc_indexes.append(doc_index)
                            group_source_urls.append(
                                doc["metadata"]["source_url"])
                            group_contents.append(doc_content)
                if not matching_docs:
                    logger.warning(
                        f"group_similar_headers: No document found for text: {text}")
            # Compute average_score and max_score
            average_score = sum(group_scores) / \
                len(group_scores) if group_scores else None
            max_score = max(group_scores) if group_scores else None
            grouped_results.append({
                "headers": group_headers,
                "contents": group_contents,
                "doc_indexes": group_doc_indexes,
                "source_urls": group_source_urls,
                "average_score": average_score,
                "max_score": max_score,
                "scores": group_scores if group_scores else None,
                "documents": group_docs
            })
        logger.info("group_similar_headers: Mapped grouped texts to documents")
    except Exception as e:
        logger.error(f"group_similar_headers: Mapping failed: {str(e)}")
        raise

    grouped_results.sort(
        key=lambda x: len(x["documents"]),
        reverse=True
    )

    return grouped_results


def filter_low_similarity_clusters(
    cluster_texts: Dict[int, List[str]],
    embed_fn: Callable[[List[str]], List[List[float]]],
    min_similarity: float
) -> Dict[int, List[str]]:
    """
    Removes clusters where the average similarity is below `min_similarity`.

    Args:
        cluster_texts (Dict[int, List[str]]): Clustered texts.
        embed_fn (Callable[[List[str]], List[List[float]]]): Function to generate embeddings.
        min_similarity (float): Minimum similarity required to keep a cluster.

    Returns:
        Dict[int, List[str]]: Filtered clusters.
    """
    filtered_clusters: Dict[int, List[str]] = {}

    for cluster_id, texts in cluster_texts.items():
        if len(texts) < 2:
            filtered_clusters[cluster_id] = texts  # Keep single-text clusters
            continue

        embeddings: np.ndarray = np.array(embed_fn(texts))
        similarity_matrix: np.ndarray = cosine_similarity(embeddings)
        avg_similarity: float = float(
            # Ignore diagonal
            np.mean(similarity_matrix[np.triu_indices(len(texts), k=1)]))

        if avg_similarity >= min_similarity:
            # Keep high-similarity clusters
            filtered_clusters[cluster_id] = texts

    return filtered_clusters


def cluster_texts(
    texts: List[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
    num_clusters: Optional[int] = None,
    *,
    min_similarity: float = 0.5  # New threshold parameter
) -> Dict[int, List[str]]:
    """
    Groups similar texts into clusters based on embeddings, filtering based on similarity.

    Args:
        texts (List[str]): List of text inputs.
        embed_fn (Callable[[List[str]], List[List[float]]]): Function to generate text embeddings.
        num_clusters (Optional[int]): Number of clusters. If None, it will be auto-determined.
        min_similarity (float): Minimum average similarity required to keep a cluster.

    Returns:
        Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of similar texts.
    """
    if not texts:
        return {}

    embeddings: List[List[float]] = embed_fn(texts)

    num_clusters = max(2, min(len(texts) // 3, 10)
                       ) if num_clusters is None else num_clusters
    # Ensure it doesn't exceed text count
    num_clusters = min(num_clusters, len(texts))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels: List[int] = kmeans.fit_predict(embeddings)

    clustered_texts: Dict[int, List[str]] = {
        i: [] for i in range(num_clusters)}
    for text, label in zip(texts, cluster_labels):
        clustered_texts[label].append(text)

    # Filter out clusters with low similarity
    filtered_clusters = filter_low_similarity_clusters(
        clustered_texts, embed_fn, min_similarity)

    return filtered_clusters


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
    embedding_function = get_embedding_function(DEFAULT_EMBED_MODEL)
    embeddings = embedding_function(texts)

    # Plot the embeddings
    plot_text_embeddings(texts, embeddings)

    print("Done")


class InfoStats(TypedDict):
    top_score: float
    avg_top_score: float
    median_score: float
    num_results: int
    avg_word_count: float
    word_diversity: float


def compute_info(results: List[SimilarityResult], top_n: int = 10) -> InfoStats:
    scores = [
        r["score"] for r in results
        if isinstance(r["score"], (int, float)) and r["score"] >= 0
    ]

    if not scores:
        return {
            "top_score": 0.0,
            "avg_top_score": 0.0,
            "median_score": 0.0,
            "num_results": 0,
            "avg_word_count": 0.0,
            "word_diversity": 0.0
        }

    # Keep the original max_score for top_score
    max_score = max(scores)

    # Use original scores for top_n calculations
    top_n_value = min(top_n, len(scores))
    top_scores = sorted(scores, reverse=True)[:top_n_value]

    # Calculate word diversity
    all_words = []
    for r in results:
        words = r["text"].lower().split()
        all_words.extend(words)

    total_words = len(all_words)
    unique_words = len(set(all_words))
    word_diversity = unique_words / total_words if total_words > 0 else 0.0

    # Use original max_score for top_score
    top_score = max_score if scores else 0.0
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    median_score = float(np.median(top_scores)) if top_scores else 0.0
    avg_word_count = sum(len(r["text"].split())
                         for r in results) / len(results)

    return {
        "top_score": top_score,
        "avg_top_score": avg_top_score,
        "median_score": median_score,
        "num_results": len(scores),
        "avg_word_count": avg_word_count,
        "word_diversity": word_diversity
    }


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

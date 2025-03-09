from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from typing import List, Dict, TypedDict


class ClassificationResult(TypedDict):
    label: str
    score: float


class SentencePairClassificationResult(TypedDict):
    label: str
    score: float
    is_continuation: bool


# Initialize the classifier once
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


def classify_sentence(sentence: str, labels: List[str]) -> List[ClassificationResult]:
    """
    Classify a sentence into one of the provided candidate labels using zero-shot classification
    and return a list of results with label, sentence, and score.

    Args:
    - sentence (str): The sentence to be classified.
    - labels (list): A list of possible labels to classify the sentence.

    Returns:
    - List[ClassificationResult]: A list of dictionaries with labels, the sentence, and their corresponding scores.
    """
    # Perform classification using the already initialized classifier
    result = classifier(sentence, labels)

    # Prepare the list of result dictionaries
    results = [
        ClassificationResult(label=label, score=score)
        for label, score in zip(result['labels'], result['scores'])
    ]

    return sorted(results, key=lambda x: x['score'], reverse=True)


def classify_sentence_pair(sentence_pair: Dict[str, str], labels: List[str] = ["Continuation/Elaboration", "Topic Shift/New Topic"]) -> List[SentencePairClassificationResult]:
    """
    Classify a pair of sentences into one of the provided candidate labels using zero-shot classification
    and return a list of results with label, sentence pair, and score, sorted by score in descending order.

    Args:
    - sentence_pair (dict): A dictionary with 'sentence1' and 'sentence2'.
    - labels (list): A list of possible labels to classify the sentence pair.

    Returns:
    - List[SentencePairClassificationResult]: A list of dictionaries with labels, the sentence pair, and their corresponding scores.
    """
    input = f"Current text:\n{sentence_pair['sentence1']}\n\nAdditional text:\n{sentence_pair['sentence2']}\n\nHow does additional text relate to the current text?"

    result = classifier(input, labels)

    # Prepare the list of result dictionaries
    results = [
        SentencePairClassificationResult(
            label=label, score=score, is_continuation=labels.index(label) == 0)
        for label, score in zip(result['labels'], result['scores'])
    ]

    return sorted(results, key=lambda x: x['score'], reverse=True)


__all__ = [
    "classify_sentence",
    "classify_sentence_pair",
]

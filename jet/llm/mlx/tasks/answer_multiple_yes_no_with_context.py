import logging
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import numpy as np
from mlx_lm.utils import TokenizerWrapper
from mlx_lm import load
import mlx.core as mx
from jet.logger import logger
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.mlx_types import LLMModelType
from typing import List, Dict, Optional, TypedDict, Literal
import pytest
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

mx.random.seed(42)


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidInputError(Exception):
    """Raised when questions or contexts are empty or invalid."""
    pass


class ClassificationError(Exception):
    """Raised when classification fails."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class QuestionContext(TypedDict):
    question: str
    context: str


class AnswerResult(TypedDict):
    question: str
    context: str
    answer: Literal["Yes", "No"]
    confidence: float
    is_valid: bool
    error: Optional[str]


class ModelComponents:
    """Encapsulates model, tokenizer, classifier, and embedder for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper, classifier: LogisticRegression, label_encoder: LabelEncoder, embedder: SentenceTransformer):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.embedder = embedder


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model, tokenizer, classifier, and embedder."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", backend="onnx")
        classifier, label_encoder = train_classifier(embedder)
        return ModelComponents(model, tokenizer, classifier, label_encoder, embedder)
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise ModelLoadError(f"Error loading model or tokenizer: {str(e)}")


def validate_inputs(questions_contexts: List[QuestionContext]) -> None:
    """Validates that questions and contexts are non-empty."""
    if not questions_contexts:
        raise InvalidInputError("Questions and contexts list cannot be empty.")
    for qc in questions_contexts:
        if not qc["question"].strip():
            raise InvalidInputError(f"Question cannot be empty: {qc}")
        if not qc["context"].strip():
            raise InvalidInputError(
                f"Context cannot be empty for question: {qc['question']}")


def embed_question_context_pairs(pairs: List[str], embedder: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Embed question-context pairs in batches using SentenceTransformer."""
    logger.info("Embedding %d question-context pairs with batch_size=%d",
                len(pairs), batch_size)
    device = "cpu"
    logger.info("Using device: %s", device)
    try:
        embeddings = embedder.encode(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=True
        )
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        logger.debug("Embeddings shape: %s, dtype: %s",
                     embeddings.shape, embeddings.dtype)
        return embeddings
    except Exception as e:
        logger.error("Error embedding pairs: %s", str(e))
        raise ClassificationError(
            f"Failed to embed question-context pairs: {str(e)}")


def train_classifier(embedder: SentenceTransformer) -> tuple[LogisticRegression, LabelEncoder]:
    """Train a logistic regression classifier on example question-context pairs."""
    logger.info("Training logistic regression classifier")
    example_pairs = [
        "Context: Venus is the second planet from the Sun and has no natural moons. Question: Does Venus have one or more moons?",
        "Context: Mars has two small moons named Phobos and Deimos. Question: Does Mars have moons?",
        "Context: Jupiter is the largest planet and has at least 79 known moons. Question: Is Jupiter moonless?",
        "Context: Saturn has 83 moons with confirmed orbits. Question: Does Saturn have moons?"
    ]
    labels = ["No", "Yes", "No", "Yes"]

    embeddings = embed_question_context_pairs(example_pairs, embedder)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    classifier = LogisticRegression(
        multi_class='ovr', solver='lbfgs', max_iter=200)
    classifier.fit(embeddings, encoded_labels)

    logger.info("Classifier trained successfully")
    return classifier, label_encoder


def answer_multiple_yes_no_with_context(
    questions_contexts: List[QuestionContext],
    model_path: LLMModelType,
    batch_size: int = 32
) -> List[AnswerResult]:
    """Answers multiple yes/no questions with context using embeddings and a classifier."""
    validate_inputs(questions_contexts)
    try:
        model_components = load_model_components(model_path)
        results = []

        # Prepare question-context pairs
        pairs = [
            f"Context: {qc['context']} Question: {qc['question']}" for qc in questions_contexts]
        logger.debug("Prepared pairs: %s", pairs)

        # Generate embeddings
        embeddings = embed_question_context_pairs(
            pairs, model_components.embedder, batch_size)

        # Classify
        pred_probas = model_components.classifier.predict_proba(embeddings)
        pred_indices = np.argmax(pred_probas, axis=1)
        confidences = pred_probas[np.arange(len(pred_indices)), pred_indices]

        for i, qc in enumerate(questions_contexts):
            try:
                predicted_label = model_components.label_encoder.inverse_transform([
                                                                                   pred_indices[i]])[0]
                confidence = float(confidences[i])
                is_valid = predicted_label in ["Yes", "No"]
                error = None if is_valid else f"Predicted label '{predicted_label}' not in ['Yes', 'No']"

                if not is_valid:
                    logger.warning(
                        "Invalid label predicted: %s for question: %s", predicted_label, qc['question'])
                    predicted_label = "No"  # Default to 'No' if invalid

                results.append(AnswerResult(
                    question=qc["question"],
                    context=qc["context"],
                    answer=predicted_label,
                    confidence=confidence,
                    is_valid=is_valid,
                    error=error
                ))
                logger.info("Question: %s, Predicted: %s, Confidence: %.4f",
                            qc["question"], predicted_label, confidence)
            except Exception as e:
                logger.error(
                    f"Error processing question '{qc['question']}': {str(e)}")
                results.append(AnswerResult(
                    question=qc["question"],
                    context=qc["context"],
                    answer="No",
                    confidence=0.0,
                    is_valid=False,
                    error=str(e)
                ))

        return results
    except (ModelLoadError, ClassificationError, InvalidInputError) as e:
        logger.error(f"Error in answer_multiple_yes_no_with_context: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    questions_contexts = [
        {"question": "Does Venus have moons?",
         "context": "Venus is the second planet from the Sun and has no natural moons."},
        {"question": "Does Mars have moons?",
         "context": "Mars has two small moons named Phobos and Deimos."},
        {"question": "Is Jupiter moonless?",
         "context": "Jupiter is the largest planet and has at least 79 known moons."}
    ]
    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

    results = answer_multiple_yes_no_with_context(
        questions_contexts, model_path)
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Context: {result['context']}")
        print(
            f"Answer: {result['answer']} (Confidence: {result['confidence']:.4f})")
        print(f"Valid: {result['is_valid']}, Error: {result['error']}\n")

# Tests


class TestAnswerMultipleYesNo:
    @pytest.fixture(autouse=True)
    def setup_components(self):
        self.model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
        self.embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", backend="onnx")
        self.classifier, self.label_encoder = train_classifier(self.embedder)

    def test_no_answer(self):
        questions_contexts = [
            {"question": "Does Venus have moons?",
             "context": "Venus is the second planet from the Sun and has no natural moons."}
        ]
        result = answer_multiple_yes_no_with_context(
            questions_contexts, self.model_path)[0]
        expected = {"answer": "No", "confidence": float,
                    "is_valid": True, "error": None}
        assert result["answer"] == expected[
            "answer"], f"Expected answer {expected['answer']}, got {result['answer']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_yes_answer(self):
        questions_contexts = [
            {"question": "Does Mars have moons?",
             "context": "Mars has two small moons named Phobos and Deimos."}
        ]
        result = answer_multiple_yes_no_with_context(
            questions_contexts, self.model_path)[0]
        expected = {"answer": "Yes", "confidence": float,
                    "is_valid": True, "error": None}
        assert result["answer"] == expected[
            "answer"], f"Expected answer {expected['answer']}, got {result['answer']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_no_answer_jupiter(self):
        questions_contexts = [
            {"question": "Is Jupiter moonless?",
             "context": "Jupiter is the largest planet and has at least 79 known moons."}
        ]
        result = answer_multiple_yes_no_with_context(
            questions_contexts, self.model_path)[0]
        expected = {"answer": "No", "confidence": float,
                    "is_valid": True, "error": None}
        assert result["answer"] == expected[
            "answer"], f"Expected answer {expected['answer']}, got {result['answer']}"
        assert isinstance(result["confidence"],
                          float), "Confidence should be a float"
        assert result["is_valid"] == expected[
            "is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected[
            "error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_invalid_input_empty_question(self):
        questions_contexts = [{"question": "", "context": "Valid context"}]
        with pytest.raises(InvalidInputError):
            answer_multiple_yes_no_with_context(
                questions_contexts, self.model_path)

    def test_invalid_input_empty_context(self):
        questions_contexts = [{"question": "Valid question", "context": ""}]
        with pytest.raises(InvalidInputError):
            answer_multiple_yes_no_with_context(
                questions_contexts, self.model_path)


class TestEmbedQuestionContextPairs:
    def test_embedding_shape(self):
        embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", backend="onnx")
        pairs = [
            "Context: Test context. Question: Test question?",
            "Context: Another context. Question: Another question?"
        ]
        result = embed_question_context_pairs(pairs, embedder)
        expected_shape = (2, embedder.get_sentence_embedding_dimension())
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_embedding_values(self):
        embedder = SentenceTransformer(
            "static-retrieval-mrl-en-v1", backend="onnx")
        pairs = ["Context: Test context. Question: Test question?"]
        result = embed_question_context_pairs(pairs, embedder)
        expected_type = np.float32
        assert result.dtype == expected_type, f"Expected dtype {expected_type}, got {result.dtype}"
        assert not np.any(np.isnan(result)), "Embeddings contain NaN values"

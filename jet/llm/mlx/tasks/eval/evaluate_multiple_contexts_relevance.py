import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import json
import pytest
from typing import List, Dict, Optional, TypedDict, Literal
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.tasks.utils import ModelComponents, load_model_components
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
import logging

mx.random.seed(42)

class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass

class InvalidInputError(Exception):
    """Raised when query or contexts are empty or invalid."""
    pass

class ClassificationError(Exception):
    """Raised when classification fails."""
    pass

class ChatMessage(TypedDict):
    role: str
    content: str

class ContextRelevanceResult(TypedDict):
    relevance_score: Literal[0, 1, 2]
    confidence: float
    is_valid: bool
    error: Optional[str]
    context: str

class ContextRelevanceResults(TypedDict):
    query: str
    results: List[ContextRelevanceResult]

class ExtendedModelComponents(ModelComponents):
    """Extends ModelComponents to include classifier, label encoder, and embedder."""
    def __init__(self, model, tokenizer, classifier: LogisticRegression, label_encoder: LabelEncoder, embedder: SentenceTransformer):
        super().__init__(model, tokenizer)
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.embedder = embedder

def load_model_components(model_path: LLMModelType) -> ExtendedModelComponents:
    """Loads model, tokenizer, classifier, and embedder."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        embedder = SentenceTransformer("static-retrieval-mrl-en-v1", backend="onnx")
        classifier, label_encoder = train_classifier(embedder)
        return ExtendedModelComponents(model, tokenizer, classifier, label_encoder, embedder)
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise ModelLoadError(f"Error loading model or tokenizer: {str(e)}")

def validate_inputs(query: str, contexts: List[str]) -> None:
    """Validates that query and contexts are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not contexts:
        raise InvalidInputError("Contexts list cannot be empty.")
    for context in contexts:
        if not context.strip():
            raise InvalidInputError(f"Context cannot be empty for query: {query}")

def embed_query_context_pairs(pairs: List[str], embedder: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Embed query-context pairs in batches using SentenceTransformer."""
    logger.info("Embedding %d query-context pairs with batch_size=%d", len(pairs), batch_size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
        logger.debug("Embeddings shape: %s, dtype: %s", embeddings.shape, embeddings.dtype)
        return embeddings
    except Exception as e:
        logger.error("Error embedding pairs: %s", str(e))
        raise ClassificationError(f"Failed to embed query-context pairs: {str(e)}")

def train_classifier(embedder: SentenceTransformer) -> tuple[LogisticRegression, LabelEncoder]:
    """Train a logistic regression classifier on example query-context pairs."""
    logger.info("Training logistic regression classifier")
    example_pairs = [
        "Query: What is the capital of France? Context: The capital of France is Paris.",
        "Query: What is the capital of France? Context: Paris is a popular tourist destination.",
        "Query: What is the capital of France? Context: Einstein developed the theory of relativity."
    ]
    labels = ["2", "1", "0"]
    
    embeddings = embed_query_context_pairs(example_pairs, embedder)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    classifier.fit(embeddings, encoded_labels)
    
    logger.info("Classifier trained successfully")
    return classifier, label_encoder

def evaluate_multiple_contexts_relevance(
    query: str,
    contexts: List[str],
    model_path: LLMModelType | ExtendedModelComponents,
    batch_size: int = 32
) -> ContextRelevanceResults:
    """Evaluates the relevance of multiple contexts for a single query using embeddings and a classifier."""
    try:
        validate_inputs(query, contexts)
        model_components = model_path if isinstance(model_path, ExtendedModelComponents) else load_model_components(model_path)
        valid_outputs = ["0", "1", "2"]
        results = []
        
        # Prepare query-context pairs
        pairs = [f"Query: {query} Context: {context}" for context in contexts]
        logger.debug("Prepared pairs: %s", pairs)
        
        # Generate embeddings
        embeddings = embed_query_context_pairs(pairs, model_components.embedder, batch_size)
        
        # Classify
        pred_probas = model_components.classifier.predict_proba(embeddings)
        pred_indices = np.argmax(pred_probas, axis=1)
        confidences = pred_probas[np.arange(len(pred_indices)), pred_indices]
        
        for i, context in enumerate(contexts):
            try:
                predicted_label = model_components.label_encoder.inverse_transform([pred_indices[i]])[0]
                confidence = float(confidences[i])
                is_valid = predicted_label in valid_outputs
                error = None if is_valid else f"Predicted label '{predicted_label}' not in {valid_outputs}"
                
                if not is_valid:
                    logger.warning("Invalid label predicted: %s for context: %s", predicted_label, context[:100])
                    predicted_label = "0"  # Default to '0' if invalid
                
                logger.info("Query: %s, Context: %s, Predicted: %s, Confidence: %.4f", 
                           query[:100], context[:100], predicted_label, confidence)
                logger.success(f"Result: {confidence:.4f}")
                
                results.append(ContextRelevanceResult(
                    context=context,
                    relevance_score=int(predicted_label),
                    confidence=confidence,
                    is_valid=is_valid,
                    error=error
                ))
            except Exception as e:
                logger.error(f"Error processing context '{context[:100]}': {str(e)}")
                results.append(ContextRelevanceResult(
                    context=context,
                    relevance_score=0,
                    confidence=0.0,
                    is_valid=False,
                    error=str(e)
                ))
        
        return ContextRelevanceResults(
            query=query,
            results=results
        )
    except (ModelLoadError, ClassificationError, InvalidInputError) as e:
        logger.error(f"Error in evaluate_multiple_contexts_relevance: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris is a popular tourist destination.",
        "Einstein developed the theory of relativity."
    ]
    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    
    result = evaluate_multiple_contexts_relevance(query, contexts, model_path)
    print(f"Query: {result['query']}")
    for res in result['results']:
        print(f"Context: {res['context']}")
        print(f"Relevance Score: {res['relevance_score']} (Confidence: {res['confidence']:.4f})")
        print(f"Valid: {res['is_valid']}, Error: {res['error']}\n")

# Tests
class TestEvaluateMultipleContextsRelevance:
    @pytest.fixture(autouse=True)
    def setup_components(self):
        self.model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
        self.embedder = SentenceTransformer("static-retrieval-mrl-en-v1", backend="onnx")
        self.classifier, self.label_encoder = train_classifier(self.embedder)

    def test_highly_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["The capital of France is Paris."]
        result = evaluate_multiple_contexts_relevance(query, contexts, self.model_path)['results'][0]
        expected = {"relevance_score": 2, "confidence": float, "is_valid": True, "error": None}
        assert result["relevance_score"] == expected["relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"], float), "Confidence should be a float"
        assert result["is_valid"] == expected["is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected["error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_somewhat_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["Paris is a popular tourist destination."]
        result = evaluate_multiple_contexts_relevance(query, contexts, self.model_path)['results'][0]
        expected = {"relevance_score": 1, "confidence": float, "is_valid": True, "error": None}
        assert result["relevance_score"] == expected["relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"], float), "Confidence should be a float"
        assert result["is_valid"] == expected["is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected["error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_not_relevant_context(self):
        query = "What is the capital of France?"
        contexts = ["Einstein developed the theory of relativity."]
        result = evaluate_multiple_contexts_relevance(query, contexts, self.model_path)['results'][0]
        expected = {"relevance_score": 0, "confidence": float, "is_valid": True, "error": None}
        assert result["relevance_score"] == expected["relevance_score"], f"Expected score {expected['relevance_score']}, got {result['relevance_score']}"
        assert isinstance(result["confidence"], float), "Confidence should be a float"
        assert result["is_valid"] == expected["is_valid"], f"Expected is_valid {expected['is_valid']}, got {result['is_valid']}"
        assert result["error"] == expected["error"], f"Expected error {expected['error']}, got {result['error']}"

    def test_invalid_input_empty_query(self):
        query = ""
        contexts = ["Valid context"]
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(query, contexts, self.model_path)

    def test_invalid_input_empty_context(self):
        query = "Valid query"
        contexts = [""]
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(query, contexts, self.model_path)

    def test_invalid_input_empty_contexts(self):
        query = "Valid query"
        contexts = []
        with pytest.raises(InvalidInputError):
            evaluate_multiple_contexts_relevance(query, contexts, self.model_path)

class TestEmbedQueryContextPairs:
    def test_embedding_shape(self):
        embedder = SentenceTransformer("static-retrieval-mrl-en-v1", backend="onnx")
        pairs = [
            "Query: Test query Context: Test context",
            "Query: Another query Context: Another context"
        ]
        result = embed_query_context_pairs(pairs, embedder)
        expected_shape = (2, embedder.get_sentence_embedding_dimension())
        assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    def test_embedding_values(self):
        embedder = SentenceTransformer("static-retrieval-mrl-en-v1", backend="onnx")
        pairs = ["Query: Test query Context: Test context"]
        result = embed_query_context_pairs(pairs, embedder)
        expected_type = np.float32
        assert result.dtype == expected_type, f"Expected dtype {expected_type}, got {result.dtype}"
        assert not np.any(np.isnan(result)), "Embeddings contain NaN values"
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pytest
import mlx.core as mx
import numpy as np
import torch
import time
import joblib
from typing import List, Dict, Optional, TypedDict, Literal, Tuple
from tqdm import tqdm
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.tasks.utils import ModelComponents, load_model_components
from jet.logger import logger
from mlx_lm import load
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

def load_model_components(model_path: LLMModelType, verbose: bool = True) -> ModelComponents:
    """Loads model and tokenizer."""
    try:
        if verbose:
            logger.debug("Loading model and tokenizer from scratch")
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise ModelLoadError(f"Error loading model or tokenizer: {str(e)}")

def load_classifier(
    example_pairs: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[LogisticRegression, LabelEncoder, SentenceTransformer]:
    """Loads or trains classifier, label encoder, and embedder, optionally from a saved directory."""
    if save_dir and os.path.exists(save_dir):
        if verbose:
            logger.info("Loading classifier components from saved directory: %s", save_dir)
        try:
            classifier = joblib.load(os.path.join(save_dir, "classifier.joblib"))
            label_encoder = joblib.load(os.path.join(save_dir, "label_encoder.joblib"))
            embedder = joblib.load(os.path.join(save_dir, "embedder.joblib"))
            logger.info(f"Classifier components loaded from {save_dir}")
            return classifier, label_encoder, embedder
        except Exception as e:
            logger.error(f"Error loading classifier components: {str(e)}")
            raise ModelLoadError(f"Failed to load classifier components: {str(e)}")
    
    try:
        if verbose:
            logger.debug("Loading embedder and training classifier from scratch")
        embedder = SentenceTransformer("static-retrieval-mrl-en-v1", device="cpu", backend="onnx")
        classifier, label_encoder = train_classifier(embedder, example_pairs, labels, verbose=verbose)
        return classifier, label_encoder, embedder
    except Exception as e:
        logger.error(f"Error loading classifier components: {str(e)}")
        raise ModelLoadError(f"Error loading classifier or embedder: {str(e)}")

def save_classifier(
    classifier: LogisticRegression,
    label_encoder: LabelEncoder,
    embedder: SentenceTransformer,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> None:
    """Save classifier, label encoder, and embedder to the specified directory."""
    if save_dir is None:
        if verbose:
            logger.info("save_dir is None, skipping classifier components save")
        return
    try:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(classifier, os.path.join(save_dir, "classifier.joblib"))
        joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))
        joblib.dump(embedder, os.path.join(save_dir, "embedder.joblib"))
        if verbose:
            logger.info(f"Classifier components saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving classifier components: {str(e)}")
        raise ModelLoadError(f"Failed to save classifier components: {str(e)}")

def validate_inputs(query: str, contexts: List[str]) -> None:
    """Validates that query and contexts are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not contexts:
        raise InvalidInputError("Contexts list cannot be empty.")
    for context in contexts:
        if not context.strip():
            raise InvalidInputError(f"Context cannot be empty for query: {query}")

def train_classifier(
    embedder: SentenceTransformer,
    example_pairs: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[LogisticRegression, LabelEncoder]:
    """Train a logistic regression classifier on example query-context pairs with progress tracking."""
    logger.info("Starting classifier training")
    start_time = time.time()
    default_pairs = [
        "Query: What is the capital of France?\nContext: The capital of France is Paris.",
        "Query: What is the capital of France?\nContext: Paris is a popular tourist destination.",
        "Query: What is the capital of France?\nContext: Einstein developed the theory of relativity.",
        "Query: List all ongoing and upcoming isekai anime 2025.\nContext: Every New Isekai Anime Announced For 2025 (So Far) By Mark Sammut Updated May 29, 2025",
        "Query: List all ongoing and upcoming isekai anime 2025.\nContext: 10 Best Isekai Anime With OP MC Reincarnated As A Child, Ranked",
        "Query: List all ongoing and upcoming isekai anime 2025.\nContext: Game Rant Advance Mario Kart World Deltarune Dune Awakening Nightreign Coming Soon"
    ]
    default_labels = ["2", "1", "0", "2", "1", "0"]
    pairs = example_pairs if example_pairs is not None else default_pairs
    labels = labels if labels is not None else default_labels
    if len(pairs) != len(labels):
        raise ValueError("Number of example pairs must match number of labels")
    logger.info(f"Processing {len(pairs)} query-context pairs")
    step_start = time.time()
    if verbose:
        logger.info("Embedding query-context pairs...")
    embeddings = embed_query_context_pairs(pairs, embedder, verbose=verbose)
    embedding_time = time.time() - step_start
    logger.info(f"Embedding completed in {embedding_time:.2f} seconds")
    step_start = time.time()
    if verbose:
        logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    label_encoding_time = time.time() - step_start
    logger.info(f"Label encoding completed in {label_encoding_time:.2f} seconds")
    step_start = time.time()
    if verbose:
        logger.info("Training logistic regression classifier...")
    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
    classifier.fit(embeddings, encoded_labels)
    training_time = time.time() - step_start
    logger.info(f"Classifier training completed in {training_time:.2f} seconds")
    total_time = time.time() - start_time
    logger.info(f"Classifier training completed successfully in {total_time:.2f} seconds")
    return classifier, label_encoder

def embed_query_context_pairs(
    pairs: List[str],
    embedder: SentenceTransformer,
    batch_size: int = 32,
    verbose: bool = True
) -> np.ndarray:
    """Embed query-context pairs in batches using SentenceTransformer with progress tracking."""
    logger.info("Embedding %d query-context pairs with batch_size=%d", len(pairs), batch_size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Using device: %s", device)
    try:
        embeddings = embedder.encode(
            pairs,
            batch_size=batch_size,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=verbose
        )
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        logger.debug("Embeddings shape: %s, dtype: %s", embeddings.shape, embeddings.dtype)
        return embeddings
    except Exception as e:
        logger.error("Error embedding pairs: %s", str(e))
        raise ClassificationError(f"Failed to embed query-context pairs: {str(e)}")

def evaluate_multiple_contexts_relevance(
    query: str,
    contexts: List[str],
    model_path: LLMModelType | ExtendedModelComponents,
    batch_size: int = 32,
    example_pairs: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> ContextRelevanceResults:
    try:
        validate_inputs(query, contexts)
        if isinstance(model_path, ExtendedModelComponents):
            model_components = model_path
        else:
            model_components = load_model_components(model_path, verbose=verbose)
            classifier, label_encoder, embedder = load_classifier(example_pairs, labels, save_dir, verbose=verbose)
            model_components = ExtendedModelComponents(
                model_components.model, model_components.tokenizer, classifier, label_encoder, embedder)
        valid_outputs = ["0", "1", "2"]
        results = []
        pairs = [f"Query: {query}\nContext: {context}" for context in contexts]
        logger.debug("Prepared pairs: %s", pairs)
        embeddings = embed_query_context_pairs(pairs, model_components.embedder, batch_size, verbose)
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
                    predicted_label = "0"
                if verbose:
                    logger.info("Query: %s\nContext: %s\nPredicted: %s\nConfidence: %.4f",
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
        return ContextRelevanceResults(query=query, results=results)
    except (ModelLoadError, ClassificationError, InvalidInputError) as e:
        logger.error(f"Error in evaluate_multiple_contexts_relevance: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris is a popular tourist destination.",
        "Einstein developed the theory of relativity."
    ]
    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    save_dir = f"{os.path.dirname(__file__)}/saved_models/default_trained"
    
    # Training step
    classifier, label_encoder, embedder = load_classifier(save_dir=save_dir, verbose=True)
    save_classifier(classifier, label_encoder, embedder, save_dir, verbose=True)
    
    # Inference step
    model_components = load_model_components(model_path, verbose=True)
    extended_components = ExtendedModelComponents(
        model_components.model, model_components.tokenizer, classifier, label_encoder, embedder)
    result = evaluate_multiple_contexts_relevance(query, contexts, extended_components, verbose=True)
    
    print(f"Query: {result['query']}")
    for res in result['results']:
        print(f"Context: {res['context']}")
        print(f"Relevance Score: {res['relevance_score']} (Confidence: {res['confidence']:.4f})")
        print(f"Valid: {res['is_valid']}, Error: {res['error']}\n")

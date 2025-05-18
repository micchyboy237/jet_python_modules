from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import TokenizerWrapper
mx.random.seed(42)


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidInputError(Exception):
    """Raised when input query or corpus is invalid."""
    pass


class RerankResult(TypedDict):
    ranked_corpus: List[str]
    scores: Dict[str, float]
    is_valid: bool
    error: Optional[str]


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def validate_inputs(query: str, corpus: List[str]) -> None:
    """Validates the query and corpus inputs."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not corpus or len(corpus) == 0:
        raise InvalidInputError("Corpus cannot be empty.")
    if any(not doc.strip() for doc in corpus):
        raise InvalidInputError("Corpus documents cannot be empty.")


def create_rerank_prompt(query: str, doc: str) -> str:
    """Creates a prompt for scoring a query-document pair."""
    return f"Evaluate the relevance of the following document to the query.\nQuery: {query}\nDocument: {doc}\nScore the relevance from 0 (irrelevant) to 1 (highly relevant)."


def log_rerank_details(query: str, corpus: List[str], model_path: LLMModelType) -> None:
    """Logs query, tokenized query, and corpus details for debugging."""
    logger.gray("Query:")
    logger.debug(query)
    logger.gray("Tokenized Query:")
    logger.debug(tokenize_strings(query, model_path))
    logger.gray("Corpus:")
    logger.debug(corpus)
    logger.newline()


def compute_relevance_score(
    model_components: ModelComponents,
    prompt: str
) -> float:
    """Computes a relevance score for a query-document pair."""
    try:
        input_ids = mx.array(model_components.tokenizer.encode(
            prompt, add_special_tokens=False))
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        model_output = model_components.model(input_ids)
        logits = model_output[0, -1]
        probs = mx.softmax(logits, axis=-1)
        # Simplified scoring: sum probabilities of tokens representing numbers 0-9
        score_tokens = model_components.tokenizer.encode(
            "0 1 2 3 4 5 6 7 8 9", add_special_tokens=False)
        score = sum(float(probs[token_id])
                    for token_id in score_tokens) / len(score_tokens)
        logger.debug(f"Computed relevance score: {score}")
        return score
    except Exception as e:
        logger.error(f"Error computing relevance score: {str(e)}")
        return 0.0


def rerank_query_corpus(
    query: str,
    corpus: List[str],
    model_path: LLMModelType,
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> RerankResult:
    """Reranks a corpus of documents based on relevance to a query."""
    try:
        validate_inputs(query, corpus)
        model_components = load_model_components(model_path)
        log_rerank_details(query, corpus, model_path)

        scores = {}
        for doc in corpus:
            prompt = create_rerank_prompt(query, doc)
            score = compute_relevance_score(model_components, prompt)
            scores[doc] = score

        # Sort corpus by scores in descending order
        ranked_corpus = sorted(corpus, key=lambda x: scores[x], reverse=True)
        logger.debug(f"Ranked corpus: {ranked_corpus}")
        logger.debug(f"Scores: {scores}")

        return RerankResult(
            ranked_corpus=ranked_corpus,
            scores=scores,
            is_valid=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in rerank_query_corpus: {str(e)}")
        return RerankResult(
            ranked_corpus=[],
            scores={},
            is_valid=False,
            error=str(e)
        )

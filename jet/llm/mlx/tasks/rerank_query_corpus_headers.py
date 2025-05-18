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
    """Raised when input query, corpus, or headers are invalid."""
    pass


class RerankResult(TypedDict):
    ranked_corpus: List[Dict[str, str]]
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


def validate_inputs(query: str, corpus: List[Dict[str, str]], headers: List[str]) -> None:
    """Validates the query, corpus, and headers inputs."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not corpus or len(corpus) == 0:
        raise InvalidInputError("Corpus cannot be empty.")
    if not headers or len(headers) == 0:
        raise InvalidInputError("Headers cannot be empty.")
    for doc in corpus:
        if not all(header in doc for header in headers):
            raise InvalidInputError(
                f"Document missing required headers: {headers}")
        if any(not doc[header].strip() for header in headers):
            raise InvalidInputError("Document fields cannot be empty.")


def create_rerank_prompt(query: str, doc: Dict[str, str], headers: List[str]) -> str:
    """Creates a prompt for scoring a query-document pair with headers."""
    doc_content = "\n".join(f"{header}: {doc[header]}" for header in headers)
    return f"Evaluate the relevance of the following document to the query.\nQuery: {query}\nDocument:\n{doc_content}\nScore the relevance from 0 (irrelevant) to 1 (highly relevant)."


def log_rerank_details(query: str, corpus: List[Dict[str, str]], headers: List[str], model_path: LLMModelType) -> None:
    """Logs query, tokenized query, and corpus details for debugging."""
    logger.gray("Query:")
    logger.debug(query)
    logger.gray("Tokenized Query:")
    logger.debug(tokenize_strings(query, model_path))
    logger.gray("Corpus (with headers):")
    logger.debug([{header: doc[header] for header in headers}
                 for doc in corpus])
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


def rerank_query_corpus_headers(
    query: str,
    corpus: List[Dict[str, str]],
    headers: List[str],
    model_path: LLMModelType,
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> RerankResult:
    """Reranks a corpus of documents with headers based on relevance to a query."""
    try:
        validate_inputs(query, corpus, headers)
        model_components = load_model_components(model_path)
        log_rerank_details(query, corpus, headers, model_path)

        scores = {}
        for doc in corpus:
            doc_key = "|".join(f"{header}:{doc[header]}" for header in headers)
            prompt = create_rerank_prompt(query, doc, headers)
            score = compute_relevance_score(model_components, prompt)
            scores[doc_key] = score

        # Sort corpus by scores in descending order
        ranked_corpus = sorted(corpus, key=lambda x: scores["|".join(
            f"{header}:{x[header]}" for header in headers)], reverse=True)
        logger.debug(f"Ranked corpus: {ranked_corpus}")
        logger.debug(f"Scores: {scores}")

        return RerankResult(
            ranked_corpus=ranked_corpus,
            scores=scores,
            is_valid=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in rerank_query_corpus_headers: {str(e)}")
        return RerankResult(
            ranked_corpus=[],
            scores={},
            is_valid=False,
            error=str(e)
        )

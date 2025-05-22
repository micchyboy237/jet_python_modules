import pytest
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
    InvalidInputError,
    InvalidOutputError
)

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_multiple_contexts_relevance():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "Paris is a popular tourist destination.",
        "Einstein developed the theory of relativity."
    ]
    results = evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)
    assert len(results) == 3
    assert results[0]["is_valid"]
    assert results[0]["query"] == query
    assert results[0]["context"] == contexts[0]
    assert results[0]["relevance_score"] == 2
    assert results[0]["error"] is None
    assert results[1]["is_valid"]
    assert results[1]["query"] == query
    assert results[1]["context"] == contexts[1]
    assert results[1]["relevance_score"] == 1
    assert results[1]["error"] is None
    assert results[2]["is_valid"]
    assert results[2]["query"] == query
    assert results[2]["context"] == contexts[2]
    assert results[2]["relevance_score"] == 0
    assert results[2]["error"] is None


def test_empty_query():
    query = ""
    contexts = ["The capital of France is Paris."]
    with pytest.raises(InvalidInputError, match="Query cannot be empty."):
        evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)


def test_empty_contexts_list():
    query = "What is the capital of France?"
    contexts = []
    with pytest.raises(InvalidInputError, match="Contexts list cannot be empty."):
        evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)


def test_empty_context():
    query = "What is the capital of France?"
    contexts = ["", "The capital of France is Paris."]
    with pytest.raises(InvalidInputError, match="Context cannot be empty"):
        evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)


def test_single_context():
    query = "What is the capital of France?"
    contexts = ["The capital of France is Paris."]
    results = evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["query"] == query
    assert results[0]["context"] == contexts[0]
    assert results[0]["relevance_score"] == 2
    assert results[0]["error"] is None


def test_mixed_valid_and_invalid_contexts():
    query = "What is the capital of France?"
    contexts = [
        "The capital of France is Paris.",
        "",
        "Paris is a popular tourist destination."
    ]
    with pytest.raises(InvalidInputError, match="Context cannot be empty"):
        evaluate_multiple_contexts_relevance(query, contexts, MODEL_PATH)


def test_custom_system_prompt():
    query = "What is the capital of France?"
    contexts = ["The capital of France is Paris."]
    custom_prompt = (
        "Evaluate the relevance of the context to the query.\n"
        "Respond with '0' (not relevant), '1' (somewhat relevant), or '2' (highly relevant).\n"
        "Example:\n"
        "Query: What is the capital of France?\n"
        "Context: The capital of France is Paris.\n"
        "Score: 2"
    )
    results = evaluate_multiple_contexts_relevance(
        query, contexts, MODEL_PATH, system_prompt=custom_prompt)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["query"] == query
    assert results[0]["context"] == contexts[0]
    assert results[0]["relevance_score"] == 2
    assert results[0]["error"] is None

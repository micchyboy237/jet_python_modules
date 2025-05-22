import pytest
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.tasks.eval.evaluate_context_relevance import evaluate_context_relevance

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_low_relevance_context():
    query = "What is the capital of France?"
    context = "The theory of relativity was developed by Albert Einstein."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 0  # Expect low
    assert result["error"] is None


def test_medium_relevance_context():
    query = "What is the capital of France?"
    context = "Paris hosts many tourists in France."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 1  # Expect medium
    assert result["error"] is None


def test_high_relevance_context():
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 2  # Expect high
    assert result["error"] is None


def test_empty_query():
    query = ""
    context = "The capital of France is Paris."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Query cannot be empty." in result["error"]


def test_empty_context():
    query = "What is the capital of France?"
    context = ""
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Context cannot be empty." in result["error"]

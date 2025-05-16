import pytest
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.eval.evaluate_context_relevance import evaluate_context_relevance

MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"


def test_relevant_context():
    query = "What is the capital of France?"
    context = "The capital city of France is Paris, located in the northern part of the country."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [3, 4]  # Expect high or very high
    assert result["error"] is None


def test_irrelevant_context():
    query = "What is the capital of France?"
    context = "The theory of relativity was developed by Albert Einstein."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [0, 1]  # Expect very low or low
    assert result["error"] is None


def test_partially_relevant_context():
    query = "What is the capital of France?"
    context = "France is a country in Europe with many cities, including Paris."
    result = evaluate_context_relevance(query, context, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [2, 3, 4]  # Expect medium to very high
    assert result["error"] is None


def test_empty_query():
    query = ""
    context = "The capital city of France is Paris."
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

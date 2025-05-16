import pytest
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.eval.evaluate_response_relevance import evaluate_response_relevance

MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"


def test_relevant_response():
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    result = evaluate_response_relevance(query, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [3, 4]  # Expect high or very high
    assert result["error"] is None


def test_irrelevant_response():
    query = "What is the capital of France?"
    response = "The sky is blue."
    result = evaluate_response_relevance(query, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [0, 1]  # Expect very low or low
    assert result["error"] is None


def test_partially_relevant_response():
    query = "What is the capital of France?"
    response = "France is in Europe, and its capital is Paris."
    result = evaluate_response_relevance(query, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] in [2, 3, 4]  # Expect medium to very high
    assert result["error"] is None


def test_empty_query():
    query = ""
    response = "The capital of France is Paris."
    result = evaluate_response_relevance(query, response, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Query cannot be empty." in result["error"]


def test_empty_response():
    query = "What is the capital of France?"
    response = ""
    result = evaluate_response_relevance(query, response, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Response cannot be empty." in result["error"]

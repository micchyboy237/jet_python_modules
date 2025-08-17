import pytest
from jet.models.model_types import LLMModelType
from jet.llm.mlx.tasks.eval.evaluate_response_relevance import evaluate_response_relevance

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_low_relevance_response():
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    response = "The theory of relativity was developed by Albert Einstein."
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 0
    assert result["error"] is None


def test_medium_relevance_response():
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    response = "Paris is a major tourist destination."
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 1
    assert result["error"] is None


def test_high_relevance_response():
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    response = "The capital of France is Paris."
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert result["is_valid"]
    assert result["relevance_score"] == 2
    assert result["error"] is None


def test_empty_query():
    query = ""
    context = "The capital of France is Paris."
    response = "The capital of France is Paris."
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Query cannot be empty." in result["error"]


def test_empty_context():
    query = "What is the capital of France?"
    context = ""
    response = "The capital of France is Paris."
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Context cannot be empty." in result["error"]


def test_empty_response():
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    response = ""
    result = evaluate_response_relevance(query, context, response, MODEL_PATH)
    assert not result["is_valid"]
    assert result["relevance_score"] == 0
    assert "Response cannot be empty." in result["error"]

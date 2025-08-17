import pytest
from jet.models.model_types import LLMModelType
from jet.llm.mlx.tasks.answer_multiple_choice import answer_multiple_choice, InvalidMethodError, InvalidOutputError, InvalidInputError

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_valid_multiple_choice():
    question = "What is the capital of France?"
    choices = ["Paris", "London", "Berlin", "Madrid"]
    result = answer_multiple_choice(question, choices, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer"] == "Paris"
    assert result["token_id"] != -1
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_empty_question():
    question = ""
    choices = ["Paris", "London", "Berlin", "Madrid"]
    result = answer_multiple_choice(question, choices, MODEL_PATH)
    assert not result["is_valid"]
    assert result["answer"] == ""
    assert result["token_id"] == -1
    assert "Question cannot be empty." in result["error"]


def test_empty_choices():
    question = "What is the capital of France?"
    choices = []
    result = answer_multiple_choice(question, choices, MODEL_PATH)
    assert not result["is_valid"]
    assert result["answer"] == ""
    assert result["token_id"] == -1
    assert "Choices cannot be empty." in result["error"]


def test_invalid_method():
    question = "What is the capital of France?"
    choices = ["Paris", "London", "Berlin", "Madrid"]
    result = answer_multiple_choice(
        question, choices, MODEL_PATH, method="invalid_method")
    assert not result["is_valid"]
    assert result["answer"] == ""
    assert result["token_id"] == -1
    assert "Invalid method specified: invalid_method" in result["error"]


def test_single_token_answer():
    question = "What is 1 + 1?"
    choices = ["2", "3", "4", "5"]
    result = answer_multiple_choice(
        question, choices, MODEL_PATH, max_tokens=1)
    assert result["is_valid"]
    assert result["answer"] == "2"
    assert result["token_id"] != -1
    assert result["error"] is None


def test_confidence_score_override():
    question = "What is the largest planet?"
    choices = ["Jupiter", "Saturn", "Earth", "Mars"]
    result = answer_multiple_choice(question, choices, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer"] == "Jupiter"
    assert result["token_id"] != -1
    assert result["error"] is None

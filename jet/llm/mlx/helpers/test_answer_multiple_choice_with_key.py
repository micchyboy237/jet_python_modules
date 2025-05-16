import pytest
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.answer_multiple_choice_with_key import answer_multiple_choice_with_key, InvalidChoiceFormatError, InvalidMethodError, InvalidInputError

MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"


def test_valid_multiple_choice():
    question = "What is the capital of France?"
    choices = ["A) Paris", "B) London", "C) Berlin", "D) Madrid"]
    result = answer_multiple_choice_with_key(question, choices, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer_key"] == "A"
    assert result["token_id"] != -1
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_invalid_choice_format():
    question = "What is the capital of France?"
    # Missing ') ' in first choice
    choices = ["A Paris", "B) London", "C) Berlin", "D) Madrid"]
    result = answer_multiple_choice_with_key(question, choices, MODEL_PATH)
    assert not result["is_valid"]
    assert result["answer_key"] == ""
    assert result["token_id"] == -1
    assert "Choice 'A Paris' does not match format 'Key) Text'" in result["error"]


def test_empty_question():
    question = ""
    choices = ["A) Paris", "B) London", "C) Berlin", "D) Madrid"]
    result = answer_multiple_choice_with_key(question, choices, MODEL_PATH)
    assert not result["is_valid"]
    assert result["answer_key"] == ""
    assert result["token_id"] == -1
    assert "Question cannot be empty." in result["error"]


def test_empty_choices():
    question = "What is the capital of France?"
    choices = []
    result = answer_multiple_choice_with_key(question, choices, MODEL_PATH)
    assert not result["is_valid"]
    assert result["answer_key"] == ""
    assert result["token_id"] == -1
    assert "Choices cannot be empty." in result["error"]


def test_invalid_method():
    question = "What is the capital of France?"
    choices = ["A) Paris", "B) London", "C) Berlin", "D) Madrid"]
    result = answer_multiple_choice_with_key(
        question, choices, MODEL_PATH, method="invalid_method")
    assert not result["is_valid"]
    assert result["answer_key"] == ""
    assert result["token_id"] == -1
    assert "Invalid method specified: invalid_method" in result["error"]


def test_single_token_answer():
    question = "What is 1 + 1?"
    choices = ["A) 2", "B) 3", "C) 4", "D) 5"]
    result = answer_multiple_choice_with_key(
        question, choices, MODEL_PATH, max_tokens=1)
    assert result["is_valid"]
    assert result["answer_key"] == "A"
    assert result["token_id"] != -1
    assert result["error"] is None


def test_confidence_score_override():
    question = "What is the largest planet?"
    choices = ["A) Jupiter", "B) Saturn", "C) Earth", "D) Mars"]
    result = answer_multiple_choice_with_key(question, choices, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer_key"] == "A"
    assert result["token_id"] != -1
    assert result["error"] is None

import pytest
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.yes_no_answer import answer_yes_no, PromptFormattingError, InvalidMethodError, InvalidOutputError

MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"


def test_valid_yes_answer():
    question = "Is the sky blue?"
    result = answer_yes_no(question, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer"] == "Yes"
    assert result["token_id"] != -1
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_valid_no_answer():
    question = "Is the sun cold?"
    result = answer_yes_no(question, MODEL_PATH)
    assert result["is_valid"]
    assert result["answer"] == "No"
    assert result["token_id"] != -1
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_empty_question():
    question = ""
    with pytest.raises(PromptFormattingError, match="Question cannot be empty."):
        answer_yes_no(question, MODEL_PATH)


def test_invalid_method():
    question = "Is the sky blue?"
    with pytest.raises(InvalidMethodError, match="Invalid method specified. Only 'generate_step' is supported."):
        answer_yes_no(question, MODEL_PATH, method="invalid_method")


def test_zero_max_tokens():
    question = "Is the sky blue?"
    with pytest.raises(ValueError, match="Max tokens can only be -1 or a positive integer."):
        answer_yes_no(question, MODEL_PATH, max_tokens=0)


def test_single_token_generation():
    question = "Is 2 + 2 equal to 4?"
    result = answer_yes_no(question, MODEL_PATH, max_tokens=1)
    assert result["is_valid"]
    assert result["answer"] == "Yes"
    assert result["token_id"] != -1
    assert result["error"] is None

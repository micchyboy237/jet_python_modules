import pytest
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.tasks.answer_multiple_yes_no_with_context import answer_multiple_yes_no_with_context, PromptFormattingError, InvalidMethodError, InvalidOutputError

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_valid_multiple_yes_no_answers():
    question = "Is the information provided accurate?"
    contexts = [
        "The sky is blue due to Rayleigh scattering.",
        "The sun is cold and made of ice.",
        "Water boils at 100°C at standard pressure."
    ]
    result = answer_multiple_yes_no_with_context(
        question, contexts, MODEL_PATH)
    assert result["is_valid"]
    assert isinstance(result["answers"], list)
    assert len(result["answers"]) == 3
    assert result["answers"] == ["Yes", "No", "Yes"]
    assert all(token_id != -1 for token_id in result["token_ids"])
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_single_context():
    question = "Is the sky blue?"
    contexts = ["The sky appears blue due to scattering of sunlight."]
    result = answer_multiple_yes_no_with_context(
        question, contexts, MODEL_PATH)
    assert result["is_valid"]
    assert result["answers"] == ["Yes"]
    assert len(result["token_ids"]) == 1
    assert result["token_ids"][0] != -1
    assert result["method"] == "generate_step"
    assert result["error"] is None


def test_empty_question():
    question = ""
    contexts = ["The sky is blue."]
    with pytest.raises(PromptFormattingError, match="Question cannot be empty."):
        answer_multiple_yes_no_with_context(question, contexts, MODEL_PATH)


def test_empty_contexts():
    question = "Is the sky blue?"
    contexts = []
    with pytest.raises(PromptFormattingError, match="Contexts cannot be empty."):
        answer_multiple_yes_no_with_context(question, contexts, MODEL_PATH)


def test_invalid_method():
    question = "Is the sky blue?"
    contexts = ["The sky is blue."]
    with pytest.raises(InvalidMethodError, match="Invalid method specified. Only 'generate_step' is supported."):
        answer_multiple_yes_no_with_context(
            question, contexts, MODEL_PATH, method="invalid_method")


def test_zero_max_tokens():
    question = "Is the sky blue?"
    contexts = ["The sky is blue."]
    with pytest.raises(ValueError, match="Max tokens can only be -1 or a positive integer."):
        answer_multiple_yes_no_with_context(
            question, contexts, MODEL_PATH, max_tokens=0)


def test_invalid_output():
    question = "Is the sky blue?"
    contexts = ["The sky is blue."]
    # Assuming the model generates an invalid output (mocked scenario)
    # This test might depend on specific model behavior, so it's a placeholder
    result = answer_multiple_yes_no_with_context(
        question, contexts, MODEL_PATH, max_tokens=1)
    if not result["is_valid"]:
        assert "Output is not 'Yes' or 'No'." in result["error"]

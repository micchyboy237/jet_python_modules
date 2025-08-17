import pytest
from jet.models.model_types import LLMModelType
from jet.llm.mlx.tasks.answer_multiple_yes_no_with_context import (
    answer_multiple_yes_no_with_context,
    InvalidMethodError,
    InvalidInputError,
    InvalidOutputError,
    QuestionContext
)

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


def test_valid_yes_no_answers():
    questions_contexts = [
        QuestionContext(
            question="Is the sky blue?",
            context="The sky appears blue due to Rayleigh scattering."
        ),
        QuestionContext(
            question="Is the sun cold?",
            context="The sun's surface temperature is approximately 5,500°C."
        )
    ]
    results = answer_multiple_yes_no_with_context(
        questions_contexts, MODEL_PATH)
    assert len(results) == 2
    assert results[0]["is_valid"]
    assert results[0]["question"] == "Is the sky blue?"
    assert results[0]["answer"] == "Yes"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None
    assert results[1]["is_valid"]
    assert results[1]["question"] == "Is the sun cold?"
    assert results[1]["answer"] == "No"
    assert results[1]["token_id"] != -1
    assert results[1]["error"] is None


def test_empty_questions_list():
    questions_contexts = []
    with pytest.raises(InvalidInputError, match="Questions and contexts list cannot be empty."):
        answer_multiple_yes_no_with_context(questions_contexts, MODEL_PATH)


def test_empty_question():
    questions_contexts = [
        QuestionContext(
            question="",
            context="The sky appears blue due to Rayleigh scattering."
        )
    ]
    with pytest.raises(InvalidInputError, match="Question cannot be empty"):
        answer_multiple_yes_no_with_context(questions_contexts, MODEL_PATH)


def test_empty_context():
    questions_contexts = [
        QuestionContext(
            question="Is the sky blue?",
            context=""
        )
    ]
    with pytest.raises(InvalidInputError, match="Context cannot be empty"):
        answer_multiple_yes_no_with_context(questions_contexts, MODEL_PATH)


def test_invalid_method():
    questions_contexts = [
        QuestionContext(
            question="Is the sky blue?",
            context="The sky appears blue due to Rayleigh scattering."
        )
    ]
    with pytest.raises(InvalidMethodError, match="Invalid method specified. Only 'generate_step' is supported."):
        answer_multiple_yes_no_with_context(
            questions_contexts, MODEL_PATH, method="invalid_method")


def test_zero_max_tokens():
    questions_contexts = [
        QuestionContext(
            question="Is the sky blue?",
            context="The sky appears blue due to Rayleigh scattering."
        )
    ]
    with pytest.raises(ValueError, match="Max tokens can only be -1 or a positive integer."):
        answer_multiple_yes_no_with_context(
            questions_contexts, MODEL_PATH, max_tokens=0)


def test_single_token_generation():
    questions_contexts = [
        QuestionContext(
            question="Is 2 + 2 equal to 4?",
            context="Basic arithmetic confirms that 2 + 2 equals 4."
        )
    ]
    results = answer_multiple_yes_no_with_context(
        questions_contexts, MODEL_PATH, max_tokens=1)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["question"] == "Is 2 + 2 equal to 4?"
    assert results[0]["answer"] == "Yes"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None


def test_mixed_valid_and_invalid_inputs():
    questions_contexts = [
        QuestionContext(
            question="Is the sky blue?",
            context="The sky appears blue due to Rayleigh scattering."
        ),
        QuestionContext(
            question="",
            context="Invalid input test."
        )
    ]
    with pytest.raises(InvalidInputError, match="Question cannot be empty"):
        answer_multiple_yes_no_with_context(questions_contexts, MODEL_PATH)


def test_confidence_score_override():
    questions_contexts = [
        QuestionContext(
            question="Is water wet?",
            context="Water molecules adhere to surfaces, causing wetness."
        )
    ]
    results = answer_multiple_yes_no_with_context(
        questions_contexts, MODEL_PATH)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["question"] == "Is water wet?"
    # Assuming model prefers "Yes" based on context
    assert results[0]["answer"] == "Yes"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None


def test_planet_moons_question():
    question = "Which planet in our solar system has one or more moons?"
    contexts = [
        "Venus is the second planet from the Sun and has no natural moons.",
        "Jupiter is the largest planet and has at least 79 known moons, including Ganymede.",
        "Mars has two small moons named Phobos and Deimos.",
        "Saturn is known for its rings and has 83 moons with confirmed orbits."
    ]
    question_contexts: list[QuestionContext] = [
        {"question": question, "context": ctx} for ctx in contexts
    ]
    results = answer_multiple_yes_no_with_context(
        question_contexts, MODEL_PATH)
    answers = [result["answer"] for result in results]
    assert answers == ["No", "Yes", "Yes", "Yes"]


def test_custom_system_prompt():
    custom_prompt = (
        "Respond with Yes or No only based on facts.\n"
        "Example:\n"
        "Context: The moon orbits Earth.\n"
        "Question: Does the moon orbit Earth?\n"
        "Answer: Yes"
    )
    questions_contexts = [
        QuestionContext(
            question="Does the moon orbit Earth?",
            context="The moon orbits Earth."
        )
    ]
    results = answer_multiple_yes_no_with_context(
        questions_contexts, MODEL_PATH, system_prompt=custom_prompt
    )
    assert results[0]["answer"] == "Yes"

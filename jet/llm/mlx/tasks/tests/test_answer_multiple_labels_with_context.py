import pytest
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.tasks.answer_multiple_labels_with_context import (
    answer_multiple_labels_with_context,
    InvalidMethodError,
    InvalidInputError,
    InvalidOutputError,
    QuestionContext
)

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"
LABELS = ["Positive", "Negative", "Neutral"]


def test_valid_label_answers():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment of the movie review?",
            context="The movie was thrilling and well-received."
        ),
        QuestionContext(
            question="What is the sentiment of the product review?",
            context="The product broke after one use."
        ),
        QuestionContext(
            question="What is the sentiment of the book review?",
            context="The book was average, neither great nor terrible."
        )
    ]
    results = answer_multiple_labels_with_context(
        questions_contexts, MODEL_PATH, LABELS)
    assert len(results) == 3
    assert results[0]["is_valid"]
    assert results[0]["question"] == "What is the sentiment of the movie review?"
    assert results[0]["answer"] == "Positive"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None
    assert results[1]["is_valid"]
    assert results[1]["question"] == "What is the sentiment of the product review?"
    assert results[1]["answer"] == "Negative"
    assert results[1]["token_id"] != -1
    assert results[1]["error"] is None
    assert results[2]["is_valid"]
    assert results[2]["question"] == "What is the sentiment of the book review?"
    assert results[2]["answer"] == "Neutral"
    assert results[2]["token_id"] != -1
    assert results[2]["error"] is None


def test_empty_questions_list():
    questions_contexts = []
    with pytest.raises(InvalidInputError, match="Questions and contexts list cannot be empty."):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS)


def test_empty_labels_list():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The movie was great."
        )
    ]
    with pytest.raises(InvalidInputError, match="Labels list cannot be empty."):
        answer_multiple_labels_with_context(questions_contexts, MODEL_PATH, [])


def test_empty_question():
    questions_contexts = [
        QuestionContext(
            question="",
            context="The movie was great."
        )
    ]
    with pytest.raises(InvalidInputError, match="Question cannot be empty"):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS)


def test_empty_context():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context=""
        )
    ]
    with pytest.raises(InvalidInputError, match="Context cannot be empty"):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS)


def test_invalid_method():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The movie was great."
        )
    ]
    with pytest.raises(InvalidMethodError, match="Invalid method specified. Only 'generate_step' is supported."):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS, method="invalid_method")


def test_zero_max_tokens():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The movie was great."
        )
    ]
    with pytest.raises(ValueError, match="Max tokens can only be -1 or a positive integer."):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS, max_tokens=0)


def test_single_token_generation():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment of the movie review?",
            context="The movie was a masterpiece."
        )
    ]
    results = answer_multiple_labels_with_context(
        questions_contexts, MODEL_PATH, LABELS, max_tokens=1)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["question"] == "What is the sentiment of the movie review?"
    assert results[0]["answer"] == "Positive"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None


def test_mixed_valid_and_invalid_inputs():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The movie was great."
        ),
        QuestionContext(
            question="",
            context="Invalid input test."
        )
    ]
    with pytest.raises(InvalidInputError, match="Question cannot be empty"):
        answer_multiple_labels_with_context(
            questions_contexts, MODEL_PATH, LABELS)


def test_confidence_score_override():
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The product was satisfactory."
        )
    ]
    results = answer_multiple_labels_with_context(
        questions_contexts, MODEL_PATH, LABELS)
    assert len(results) == 1
    assert results[0]["is_valid"]
    assert results[0]["question"] == "What is the sentiment?"
    assert results[0]["answer"] == "Neutral"
    assert results[0]["token_id"] != -1
    assert results[0]["error"] is None


def test_sentiment_analysis():
    question = "What is the sentiment of the review?"
    contexts = [
        "The movie was a thrilling adventure with stunning visuals.",
        "The product failed to meet expectations and broke quickly.",
        "The service was adequate but unremarkable."
    ]
    question_contexts: list[QuestionContext] = [
        {"question": question, "context": ctx} for ctx in contexts
    ]
    results = answer_multiple_labels_with_context(
        question_contexts, MODEL_PATH, LABELS)
    answers = [result["answer"] for result in results]
    assert answers == ["Positive", "Negative", "Neutral"]


def test_custom_system_prompt():
    custom_prompt = (
        "Respond with one of 'Positive', 'Negative', or 'Neutral' based on facts.\n"
        "Example:\n"
        "Context: The service was exceptional.\n"
        "Question: What is the sentiment?\n"
        "Answer: Positive"
    )
    questions_contexts = [
        QuestionContext(
            question="What is the sentiment?",
            context="The service was exceptional."
        )
    ]
    results = answer_multiple_labels_with_context(
        questions_contexts, MODEL_PATH, LABELS, system_prompt=custom_prompt
    )
    assert results[0]["answer"] == "Positive"

import pytest
from typing import Iterator, Dict, Any
from unittest.mock import patch
from jet.llm.evaluators.context_relevancy_evaluator import (
    EvaluationComment,
    EvaluationResult,
    EvaluationDetails,
    evaluate_context_relevancy,
    EVAL_QUESTIONS
)


@pytest.fixture
def mock_stream_chat_low_score():
    """Mock stream_chat to return a low score response."""
    def mock_stream_chat_side_effect(messages, **kwargs) -> Iterator[Dict[str, Any]]:
        response = (
            "Feedback:\n"
            "Q1: NO - Context is unrelated to Python programming. (Score: 0)\n"
            "Q2: NO - Context cannot answer the query about Python. (Score: 0)\n\n"
            "[RESULT] 0.0\n"
        )
        yield {"choices": [{"message": {"content": response}}]}
    with patch("jet.llm.evaluators.context_relevancy_evaluator.stream_chat") as mock:
        mock.side_effect = mock_stream_chat_side_effect
        yield mock


@pytest.fixture
def mock_stream_chat_medium_score():
    """Mock stream_chat to return a medium score response."""
    def mock_stream_chat_side_effect(messages, **kwargs) -> Iterator[Dict[str, Any]]:
        response = (
            "Feedback:\n"
            "Q1: YES - Context mentions Python but not decorators specifically. (Score: 1)\n"
            "Q2: NO - Context does not provide details about decorators. (Score: 0)\n\n"
            "[RESULT] 1.0\n"
        )
        yield {"choices": [{"message": {"content": response}}]}
    with patch("jet.llm.evaluators.context_relevancy_evaluator.stream_chat") as mock:
        mock.side_effect = mock_stream_chat_side_effect
        yield mock


@pytest.fixture
def mock_stream_chat_high_score():
    """Mock stream_chat to return a high score response."""
    def mock_stream_chat_side_effect(messages, **kwargs) -> Iterator[Dict[str, Any]]:
        response = (
            "Feedback:\n"
            "Q1: YES - Context directly discusses Python decorators. (Score: 2)\n"
            "Q2: YES - Context fully explains how to use decorators. (Score: 2)\n\n"
            "[RESULT] 4.0\n"
        )
        yield {"choices": [{"message": {"content": response}}]}
    with patch("jet.llm.evaluators.context_relevancy_evaluator.stream_chat") as mock:
        mock.side_effect = mock_stream_chat_side_effect
        yield mock


class TestContextRelevancyEvaluatorLowScore:
    """Tests for low context relevancy score scenario."""

    def test_evaluate_context_relevancy_low_score(self, mock_stream_chat_low_score):
        # Given
        query = "What are Python decorators?"
        contexts = ["The history of Java programming."]
        expected_score = 0.0
        expected_passing = False
        expected_comments = [
            EvaluationComment(
                score=0.0,
                explanation="Context is unrelated to Python programming.",
                question=EVAL_QUESTIONS[0],
                answer="NO"
            ),
            EvaluationComment(
                score=0.0,
                explanation="Context cannot answer the query about Python.",
                question=EVAL_QUESTIONS[1],
                answer="NO"
            )
        ]
        expected_excerpts = []

        # When
        result = evaluate_context_relevancy(query=query, contexts=contexts)

        # Then
        assert result.score == expected_score
        assert result.passing == expected_passing
        assert result.details.comments == expected_comments
        assert result.excerpts == expected_excerpts


class TestContextRelevancyEvaluatorMediumScore:
    """Tests for medium context relevancy score scenario."""

    def test_evaluate_context_relevancy_medium_score(self, mock_stream_chat_medium_score):
        # Given
        query = "What are Python decorators?"
        contexts = ["Python is a versatile programming language."]
        expected_score = 1.0
        expected_passing = False
        expected_comments = [
            EvaluationComment(
                score=1.0,
                explanation="Context mentions Python but not decorators specifically.",
                question=EVAL_QUESTIONS[0],
                answer="YES"
            ),
            EvaluationComment(
                score=0.0,
                explanation="Context does not provide details about decorators.",
                question=EVAL_QUESTIONS[1],
                answer="NO"
            )
        ]
        expected_excerpts = []

        # When
        result = evaluate_context_relevancy(query=query, contexts=contexts)

        # Then
        assert result.score == expected_score
        assert result.passing == expected_passing
        assert result.details.comments == expected_comments
        assert result.excerpts == expected_excerpts


class TestContextRelevancyEvaluatorHighScore:
    """Tests for high context relevancy score scenario."""

    def test_evaluate_context_relevancy_high_score(self, mock_stream_chat_high_score):
        # Given
        query = "What are Python decorators?"
        contexts = [
            "Python decorators are a way to modify functions using other functions."]
        expected_score = 4.0
        expected_passing = True
        expected_comments = [
            EvaluationComment(
                score=2.0,
                explanation="Context directly discusses Python decorators.",
                question=EVAL_QUESTIONS[0],
                answer="YES"
            ),
            EvaluationComment(
                score=2.0,
                explanation="Context fully explains how to use decorators.",
                question=EVAL_QUESTIONS[1],
                answer="YES"
            )
        ]
        expected_excerpts = [
            "Q1: YES - Context directly discusses Python decorators. (Score: 2)",
            "Q2: YES - Context fully explains how to use decorators. (Score: 2)"
        ]

        # When
        result = evaluate_context_relevancy(query=query, contexts=contexts)

        # Then
        assert result.score == expected_score
        assert result.passing == expected_passing
        assert result.details.comments == expected_comments
        assert result.excerpts == expected_excerpts

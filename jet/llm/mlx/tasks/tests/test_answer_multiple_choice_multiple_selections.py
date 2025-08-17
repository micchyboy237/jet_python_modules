from jet.models.model_types import LLMModelType
import pytest
from typing import List
from jet.llm.mlx.tasks.answer_multiple_choice_multiple_selections import answer_multiple_choice_multiple_selections, AnswerResult, InvalidChoiceFormatError, InvalidInputError, InvalidOutputError, validate_answer

MODEL_PATH: LLMModelType = "llama-3.2-3b-instruct-4bit"


class TestMultipleChoiceWithKey:
    def setup_method(self):
        # Replace with your model path
        self.question = "Which colors are in the rainbow?"
        self.choices = ["A) Red", "B) Blue", "C) Yellow", "D) Black"]

    def test_multiple_selection(self):
        expected_answer_keys = ["A", "B", "C"]
        expected_texts = ["Red", "Blue", "Yellow"]
        expected_is_valid = True
        expected_error = None
        expected_prob = float  # Prob should be a float >= 0

        result = answer_multiple_choice_multiple_selections(
            question=self.question,
            choices=self.choices,
            model_path=MODEL_PATH,
            max_selections=3
        )

        assert result["answer_keys"] == expected_answer_keys
        assert result["texts"] == expected_texts
        assert result["is_valid"] == expected_is_valid
        assert result["error"] == expected_error
        assert isinstance(result["prob"], expected_prob)
        assert result["prob"] >= 0.0

    def test_single_selection(self):
        expected_answer_keys = ["A"]
        expected_texts = ["Red"]
        expected_is_valid = True
        expected_error = None
        expected_prob = float

        result = answer_multiple_choice_multiple_selections(
            question="Which color is primary?",
            choices=self.choices,
            model_path=MODEL_PATH,
            max_selections=1
        )

        assert result["answer_keys"] == expected_answer_keys
        assert result["texts"] == expected_texts
        assert result["is_valid"] == expected_is_valid
        assert result["error"] == expected_error
        assert isinstance(result["prob"], expected_prob)
        assert result["prob"] >= 0.0

    def test_invalid_choice_format(self):
        invalid_choices = ["A) Red", "B Blue", "C) Yellow"]
        expected_error = "Choice 'B Blue' does not match expected format"

        with pytest.raises(InvalidChoiceFormatError) as exc_info:
            answer_multiple_choice_multiple_selections(
                question=self.question,
                choices=invalid_choices,
                model_path=MODEL_PATH
            )

        result = str(exc_info.value)
        assert expected_error in result

    def test_invalid_output(self):
        # Mock scenario where model outputs invalid choice
        with pytest.raises(InvalidOutputError) as exc_info:
            # Simulate invalid output by manipulating validate_answer
            validate_answer(["Green"], ["Red", "Blue", "Yellow", "Black"])

        result = str(exc_info.value)
        expected_error = "Output 'Green' is not one of the provided choices"
        assert expected_error in result

    def test_empty_question(self):
        expected_error = "Question cannot be empty"

        with pytest.raises(InvalidInputError) as exc_info:
            answer_multiple_choice_multiple_selections(
                question="",
                choices=self.choices,
                model_path=MODEL_PATH
            )

        result = str(exc_info.value)
        assert expected_error in result

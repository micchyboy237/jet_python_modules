import unittest
import os
from jet.llm.mlx.helpers.yes_no_answer import answer_yes_no, AnswerResult, ModelLoadError, InvalidMethodError, PromptFormattingError, TokenEncodingError, GenerationError, InvalidOutputError
from jet.llm.mlx.mlx_types import ModelType


class TestYesNoAnswer(unittest.TestCase):
    def setUp(self):
        # Define a lightweight model for testing
        self.model: ModelType = "llama-3.2-3b-instruct-4bit"
        self.valid_question = "Is 1 + 1 equal to 2?"
        self.max_tokens = 1
        self.temperature = 0.1
        self.top_p = 0.1

    def test_invalid_method(self):
        """Test that an invalid method raises InvalidMethodError."""
        with self.assertRaises(InvalidMethodError):
            answer_yes_no(self.valid_question, self.model,
                          method="invalid_method")

    def test_empty_question(self):
        """Test that an empty question is handled (depends on tokenizer behavior)."""
        with self.assertRaises(PromptFormattingError) as cm:
            answer_yes_no("", self.model, method="stream_generate")

        expected = "Question cannot be empty."
        result = str(cm.exception)
        self.assertEqual(result, expected)

    def test_valid_question_stream_generate(self):
        """Test successful execution with stream_generate method."""
        try:
            result = answer_yes_no(
                self.valid_question,
                self.model,
                method="stream_generate",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            self.assertTrue(result["is_valid"], f"Error: {result['error']}")
            self.assertEqual(result["answer"].lower(), "yes")
            self.assertIsInstance(result["token_id"], int)
            self.assertGreaterEqual(result["token_id"], 0)
            self.assertEqual(result["method"], "stream_generate")
            self.assertIsNone(result["error"])
        except ModelLoadError as e:
            self.skipTest(f"Model could not be loaded: {e}")

    def test_valid_question_generate_step(self):
        """Test successful execution with generate_step method."""
        result = answer_yes_no(
            self.valid_question,
            self.model,
            method="generate_step",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        self.assertTrue(result["is_valid"], f"Error: {result['error']}")
        self.assertEqual(result["answer"].lower(), "yes")
        self.assertIsInstance(result["token_id"], int)
        self.assertGreaterEqual(result["token_id"], 0)
        self.assertEqual(result["method"], "generate_step")
        self.assertIsNone(result["error"])

    def test_invalid_model_path(self):
        """Test that an invalid model path raises ModelLoadError."""
        with self.assertRaises(ValueError):
            answer_yes_no(self.valid_question, model_path="invalid/model/path")

    def test_max_tokens_zero(self):
        """Test that max_tokens=0 is handled (may raise GenerationError or return invalid result)."""
        with self.assertRaises(ValueError):
            result = answer_yes_no(
                self.valid_question,
                self.model,
                method="stream_generate",
                max_tokens=0
            )

    def test_temperature_extreme(self):
        """Test extreme temperature value."""
        try:
            result = answer_yes_no(
                self.valid_question,
                self.model,
                method="stream_generate",
                temperature=0.0  # Extreme but valid
            )
            self.assertTrue(result["is_valid"], f"Error: {result['error']}")
            self.assertIn(result["answer"].lower(), ["yes", "no"])
        except ModelLoadError as e:
            self.skipTest(f"Model could not be loaded: {e}")

    def test_top_p_extreme(self):
        """Test extreme top_p value."""
        try:
            result = answer_yes_no(
                self.valid_question,
                self.model,
                method="stream_generate",
                top_p=1.0  # Extreme but valid
            )
            self.assertTrue(result["is_valid"], f"Error: {result['error']}")
            self.assertIn(result["answer"].lower(), ["yes", "no"])
        except ModelLoadError as e:
            self.skipTest(f"Model could not be loaded: {e}")


if __name__ == "__main__":
    unittest.main()

import unittest
from jet.llm.mlx.helpers.answer_multiple_choice import (
    answer_multiple_choice,
    create_system_prompt,
    format_chat_messages,
    validate_answer,
    InvalidOutputError,
)
from jet.llm.mlx.mlx_types import ModelType

# Test data
MODEL_PATH: ModelType = "llama-3.2-3b-instruct-4bit"
QUESTION = "Which planet is known as the Red Planet?"
CHOICES = ["Mars", "Earth", "Jupiter", "Saturn"]


class TestAnswerMultipleChoice(unittest.TestCase):
    def test_create_system_prompt(self):
        """Test system prompt creation with choices."""
        expected_prompt = (
            "Answer the following question by choosing one of the options provided "
            "without any additional text.\nOptions:\nMars\nEarth\nJupiter\nSaturn"
        )
        result = create_system_prompt(CHOICES)
        self.assertEqual(result, expected_prompt)

    def test_format_chat_messages(self):
        """Test formatting of chat messages."""
        system_prompt = create_system_prompt(CHOICES)
        messages = format_chat_messages(system_prompt, QUESTION)
        expected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTION}
        ]
        self.assertEqual(messages, expected_messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_validate_answer_valid(self):
        """Test validate_answer with a valid choice."""
        try:
            validate_answer("Mars", CHOICES)
        except InvalidOutputError:
            self.fail("Valid answer raised InvalidOutputError")

    def test_validate_answer_invalid(self):
        """Test validate_answer with an invalid choice."""
        with self.assertRaises(InvalidOutputError) as context:
            validate_answer("Venus", CHOICES)
        self.assertIn(
            "Output 'Venus' is not one of the provided choices", str(context.exception))

    def test_answer_multiple_choice_stream_generate(self):
        """Test answer_multiple_choice with stream_generate method."""
        result = answer_multiple_choice(
            question=QUESTION,
            choices=CHOICES,
            model_path=MODEL_PATH,
            method="stream_generate",
            max_tokens=10,
            temperature=0.0,
            top_p=0.9
        )
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["answer"], "Mars")
        self.assertEqual(result["method"], "stream_generate")
        self.assertIsNone(result["error"])

    def test_answer_multiple_choice_generate_step(self):
        """Test answer_multiple_choice with generate_step method."""
        result = answer_multiple_choice(
            question=QUESTION,
            choices=CHOICES,
            model_path=MODEL_PATH,
            method="generate_step",
            max_tokens=10,
            temperature=0.0,
            top_p=0.9
        )
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["answer"], "Mars")
        self.assertEqual(result["method"], "generate_step")
        self.assertIsNone(result["error"])

    def test_answer_multiple_choice_invalid_method(self):
        """Test answer_multiple_choice with an invalid method."""
        result = answer_multiple_choice(
            question=QUESTION,
            choices=CHOICES,
            model_path=MODEL_PATH,
            method="invalid_method"
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["token_id"], -1)
        self.assertIn("Invalid method specified", result["error"])

    def test_answer_multiple_choice_invalid_model(self):
        """Test answer_multiple_choice with an invalid model path."""
        result = answer_multiple_choice(
            question=QUESTION,
            choices=CHOICES,
            model_path="invalid/model/path"
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["token_id"], -1)
        self.assertIn("Error loading model or tokenizer", result["error"])


if __name__ == '__main__':
    unittest.main()

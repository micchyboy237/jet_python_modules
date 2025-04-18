import unittest
from jet.llm.evaluators.helpers.base import parse_feedback, Comment, EvaluationDetails


class TestParseFeedback(unittest.TestCase):
    def test_valid_feedback(self):
        sample = """Feedback:
1. The response matches the subject matter by providing the correct answer to the user's question about the capital of France. (Score: 1.0)
2. The response addresses the focus by giving a clear and accurate answer regarding the capital city. (Score: 0.9)

[RESULT] 1.9"""
        expected = EvaluationDetails(comments=[
            Comment(
                text="1. The response matches the subject matter by providing the correct answer to the user's question about the capital of France.",
                score=1.0
            ),
            Comment(
                text="2. The response addresses the focus by giving a clear and accurate answer regarding the capital city.",
                score=0.9
            )
        ])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_empty_feedback(self):
        sample = ""
        expected = EvaluationDetails()
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_none_feedback(self):
        sample = None
        expected = EvaluationDetails()
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())


if __name__ == "__main__":
    unittest.main()

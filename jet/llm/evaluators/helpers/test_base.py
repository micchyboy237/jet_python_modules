from jet.llm.evaluators.helpers.base import parse_feedback, Comment, EvaluationDetails
import unittest


class TestParseFeedback(unittest.TestCase):
    def test_valid_feedback(self):
        sample = """Feedback:
Q1: YES - The response clearly addresses the topic of the query about the capital of France. (Score: 1.0)
Q2: NO - The response does not provide additional details requested in the query. (Score: 0.0)

[RESULT] 1.0"""
        expected = EvaluationDetails(comments=[
            Comment(
                text="Q1: YES - The response clearly addresses the topic of the query about the capital of France.",
                score=1.0,
                passing=True
            ),
            Comment(
                text="Q2: NO - The response does not provide additional details requested in the query.",
                score=0.0,
                passing=False
            )
        ])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_empty_feedback(self):
        sample = ""
        expected = EvaluationDetails(comments=[])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_none_feedback(self):
        sample = None
        expected = EvaluationDetails(comments=[])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_invalid_feedback(self):
        sample = """Feedback:
Q1: YES - The response is relevant. (Score: invalid)
Q2: NO - The response lacks focus. (Score: 0.5)

[RESULT] 0.5"""
        expected = EvaluationDetails(comments=[
            Comment(
                text="Q2: NO - The response lacks focus.",
                score=0.5,
                passing=True
            )
        ])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_partial_feedback(self):
        sample = """Feedback:
Q1: YES - The response matches the subject matter. (Score: 0.7)
Q2: YES - The response partially addresses the focus. (Score: 0.5)

[RESULT] 1.2"""
        expected = EvaluationDetails(comments=[
            Comment(
                text="Q1: YES - The response matches the subject matter.",
                score=0.7,
                passing=True
            ),
            Comment(
                text="Q2: YES - The response partially addresses the focus.",
                score=0.5,
                passing=True
            )
        ])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())

    def test_feedback_without_prefix(self):
        sample = """Q1: YES - The response directly answers the question about the capital of France. (Score: 1.0)
Q2: YES - The response provides a specific answer, aligning with the user's request for information about the capital. (Score: 1.0)

[RESULT] 2.0"""
        expected = EvaluationDetails(comments=[
            Comment(
                text="Q1: YES - The response directly answers the question about the capital of France.",
                score=1.0,
                passing=True
            ),
            Comment(
                text="Q2: YES - The response provides a specific answer, aligning with the user's request for information about the capital.",
                score=1.0,
                passing=True
            )
        ])
        result = parse_feedback(sample)
        self.assertEqual(result.model_dump(), expected.model_dump())


if __name__ == "__main__":
    unittest.main()

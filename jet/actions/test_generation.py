import unittest
from unittest.mock import patch
from jet.actions.generation import call_ollama_generate


class TestCallOllamaGenerate(unittest.TestCase):
    @patch("jet.llm.call_ollama.requests.post")
    def test_call_ollama_generate_success(self, mock_post):
        sample = "What is 2 + 2?"
        expected = {"response": "4"}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = expected
        mock_post.return_value.iter_lines.return_value = [
            b'{"response": "4", "done": true}']

        result = call_ollama_generate(sample, stream=False)
        self.assertEqual(result, expected["response"])

    @patch("jet.llm.call_ollama.requests.post")
    def test_call_ollama_generate_stream(self, mock_post):
        sample = "Tell me a joke"
        expected = {
            "response": "Why did the chicken cross the road?", "done": False}
        mock_post.return_value.iter_lines.return_value = [
            b'{"response": "Why did the chicken cross the road?", "done": false}',
            b'{"response": "To get to the other side.", "done": true}'
        ]

        results = []
        for chunk in call_ollama_generate(sample, stream=True):
            results.append(chunk)
        self.assertIn("Why did the chicken", "".join(results))


if __name__ == '__main__':
    unittest.main()

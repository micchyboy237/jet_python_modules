import unittest
from unittest.mock import MagicMock

from jet.token.token_utils import split_texts

# Mock dependencies
OLLAMA_MODEL_EMBEDDING_TOKENS = {
    "test-model": 100  # Default token limit for testing
}


def get_tokenizer(model):
    return MagicMock(
        # Mock encoding (1 char = 1 token)
        encode=lambda text: list(range(len(text))),
        decode=lambda tokens, skip_special_tokens=True: "".join(
            chr(97 + (t % 26)) for t in tokens)  # Mock decoding
    )


# Import the function to test (assuming it's in the same module)
class TestSplitTexts(unittest.TestCase):

    def setUp(self):
        self.tokenizer = get_tokenizer("test-model")

    def test_single_text_no_splitting_needed(self):
        text = "hello world"
        result = split_texts(text, self.tokenizer, chunk_size=50)
        self.assertEqual(result, [text])

    def test_single_text_with_chunking(self):
        text = "abcdefghij" * 10  # 100 characters (tokens)
        result = split_texts(text, self.tokenizer,
                             chunk_size=30, chunk_overlap=5)
        self.assertTrue(len(result) > 1)  # Should be split
        self.assertEqual(result[1][-5:], result[2][:5])  # Overlap check

    def test_multiple_texts(self):
        # Two texts, each 100 tokens
        texts = ["abcdefghij" * 10, "klmnopqrst" * 10]
        result = split_texts(texts, self.tokenizer,
                             chunk_size=30, chunk_overlap=5)
        self.assertTrue(len(result) > 2)  # Should split both texts

    def test_chunks_do_not_exceed_max_length(self):
        text = "abcdefghij" * 20  # 200 characters (tokens)
        chunk_size = 50
        buffer = 5
        chunk_overlap = 10

        result = split_texts(text, self.tokenizer, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, buffer=buffer)

        effective_max_tokens = chunk_size - buffer
        for chunk in result:
            tokens = self.tokenizer.encode(chunk)
            self.assertLessEqual(len(tokens), effective_max_tokens,
                                 f"Chunk exceeded max length: {len(tokens)} > {effective_max_tokens}")

    def test_chunk_size_smaller_than_overlap(self):
        with self.assertRaises(ValueError) as context:
            split_texts("test text", self.tokenizer,
                        chunk_size=10, chunk_overlap=15)
        self.assertIn("must be greater than chunk overlap",
                      str(context.exception))

    def test_effective_max_tokens_too_small(self):
        with self.assertRaises(ValueError) as context:
            split_texts("test text", self.tokenizer,
                        chunk_size=10, chunk_overlap=5, buffer=6)
        self.assertIn("Effective max tokens", str(context.exception))

    def test_empty_string_input(self):
        result = split_texts("", self.tokenizer, chunk_size=10)
        self.assertEqual(result, [""])

    def test_empty_list_input(self):
        result = split_texts([], self.tokenizer, chunk_size=10)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

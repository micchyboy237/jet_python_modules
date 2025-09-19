import unittest
from unittest.mock import MagicMock

from jet.llm.models import OLLAMA_LLM_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS
from jet._token.token_utils import get_model_by_max_predict, group_texts, split_texts, token_counter


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


class TestGroupTextsOptimized(unittest.TestCase):

    def test_empty_input(self):
        result = group_texts([], "mistral", 100)
        self.assertEqual(result, [])

    def test_single_short_text(self):
        texts = ["Hello world!"]
        result = group_texts(texts, "mistral", 100)
        self.assertEqual(result, [["Hello world!"]])

    def test_multiple_short_texts(self):
        texts = ["Text 1", "Text 2", "Text 3"]
        max_tokens = 50
        token_counts = token_counter(texts, "mistral", prevent_total=True)
        result = group_texts(texts, "mistral", max_tokens)
        expected_groups = []
        current_group = []
        current_token_count = 0

        for text, token_count in zip(texts, token_counts):
            if current_token_count + token_count > max_tokens:
                expected_groups.append(current_group)
                current_group = []
                current_token_count = 0
            current_group.append(text)
            current_token_count += token_count
        if current_group:
            expected_groups.append(current_group)

        self.assertEqual(result, expected_groups)

    def test_large_text_splits_correctly(self):
        texts = ["Large text 1", "Small text", "Large text 2", "Tiny"]
        max_tokens = 100
        token_counts = token_counter(texts, "mistral", prevent_total=True)
        result = group_texts(texts, "mistral", max_tokens)

        expected_groups = []
        current_group = []
        current_token_count = 0

        for text, token_count in zip(texts, token_counts):
            if current_token_count + token_count > max_tokens:
                expected_groups.append(current_group)
                current_group = []
                current_token_count = 0
            current_group.append(text)
            current_token_count += token_count
        if current_group:
            expected_groups.append(current_group)

        self.assertEqual(result, expected_groups)

    def test_edge_case_exact_fit(self):
        texts = ["Text A", "Text B", "Text C"]
        max_tokens = 100
        token_counts = token_counter(texts, "mistral", prevent_total=True)
        result = group_texts(texts, "mistral", max_tokens)

        expected_groups = []
        current_group = []
        current_token_count = 0

        for text, token_count in zip(texts, token_counts):
            if current_token_count + token_count > max_tokens:
                expected_groups.append(current_group)
                current_group = []
                current_token_count = 0
            current_group.append(text)
            current_token_count += token_count
        if current_group:
            expected_groups.append(current_group)

        self.assertEqual(result, expected_groups)

    def test_edge_case_max_token(self):
        texts = ["Exact max token text"]
        max_tokens = 100
        token_counts = token_counter(texts, "mistral", prevent_total=True)
        result = group_texts(texts, "mistral", max_tokens)

        expected_groups = []
        current_group = []
        current_token_count = 0

        for text, token_count in zip(texts, token_counts):
            if current_token_count + token_count > max_tokens:
                expected_groups.append(current_group)
                current_group = []
                current_token_count = 0
            current_group.append(text)
            current_token_count += token_count
        if current_group:
            expected_groups.append(current_group)

        self.assertEqual(result, expected_groups)


class TestGetModelByMaxPredict(unittest.TestCase):

    def test_short_text_fits_all(self):
        sample = "Hello world!"
        max_predict = 100
        # Safely select the model with the smallest embedding token limit
        valid_models = {
            model: tokens
            for model, tokens in OLLAMA_MODEL_EMBEDDING_TOKENS.items()
            if model in OLLAMA_LLM_MODELS.__args__
        }

        if not valid_models:
            raise ValueError(
                "No valid models found in both OLLAMA_MODEL_EMBEDDING_TOKENS and OLLAMA_LLM_MODELS.")

        # Choose the model with the minimum token limit
        expected = min(valid_models.items(), key=lambda item: item[1])[0]
        result = get_model_by_max_predict(sample, max_predict)
        self.assertEqual(result, expected)

    def test_text_needs_larger_model(self):
        sample = "This is a longer text. " * 100
        max_predict = 1000
        result = get_model_by_max_predict(sample, max_predict)
        sample_tokens = token_counter(sample)
        self.assertTrue(sample_tokens + max_predict <=
                        OLLAMA_MODEL_EMBEDDING_TOKENS[result])

    def test_no_model_fits(self):
        sample = "a " * 100_000
        max_predict = 5000
        with self.assertRaises(ValueError):
            get_model_by_max_predict(sample, max_predict)


if __name__ == "__main__":
    unittest.main()

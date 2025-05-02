from jet.llm.mlx.token_utils import merge_texts
from mlx_lm import load
import nltk
import unittest

_, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")


class TestMergeTexts(unittest.TestCase):
    def setUp(self):
        # Use the actual tokenizer
        self.tokenizer = tokenizer
        # Ensure NLTK sentence tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def test_no_max_length(self):
        text = "Simple sentence. Another sentence here."
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=None)

        # Tokenize to get expected token count
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        decoded_tokens = self.tokenizer.batch_decode(
            [[tid] for tid in tokens], skip_special_tokens=True
        )

        self.assertEqual(len(result["tokens"]), len(tokens))
        self.assertEqual(result["decoded_tokens"], decoded_tokens)
        self.assertEqual(result["texts"], nltk.sent_tokenize(text))
        self.assertEqual(result["metadata"]["total_tokens"], len(tokens))
        self.assertFalse(result["metadata"]["is_truncated"])

    def test_max_length_exact_fit(self):
        text = "Simple sentence."
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=5)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:5]
        decoded_tokens = self.tokenizer.batch_decode(
            [[tid] for tid in tokens], skip_special_tokens=True
        )

        self.assertEqual(len(result["tokens"]), min(5, len(tokens)))
        self.assertEqual(result["decoded_tokens"], decoded_tokens)
        self.assertEqual(result["texts"], [text])
        self.assertEqual(result["metadata"]["total_tokens"], len(
            self.tokenizer.encode(text, add_special_tokens=False)))
        self.assertFalse(result["metadata"]["is_truncated"]
                         if len(tokens) <= 5 else True)

    def test_max_length_truncation(self):
        text = "Simple sentence. Another sentence here."
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=5)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:5]
        decoded_tokens = self.tokenizer.batch_decode(
            [[tid] for tid in tokens], skip_special_tokens=True
        )
        sentences = nltk.sent_tokenize(text)
        expected_texts = [s for s in sentences if len(
            self.tokenizer.encode(s, add_special_tokens=False)) <= 5]

        self.assertEqual(len(result["tokens"]), 4)
        self.assertEqual(result["decoded_tokens"], decoded_tokens)
        self.assertEqual(result["texts"], expected_texts[:1])
        self.assertEqual(result["metadata"]["total_tokens"], len(
            self.tokenizer.encode(text, add_special_tokens=False)))
        self.assertTrue(result["metadata"]["is_truncated"])

    def test_sentence_merging(self):
        text = "Short. Combined short sentences."
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=2)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:2]
        decoded_tokens = self.tokenizer.batch_decode(
            [[tid] for tid in tokens], skip_special_tokens=True
        )

        self.assertEqual(len(result["tokens"]), min(2, len(tokens)))
        self.assertEqual(result["decoded_tokens"], decoded_tokens)
        self.assertEqual(result["texts"], nltk.sent_tokenize(text)[:1])
        self.assertEqual(result["metadata"]["total_tokens"], len(
            self.tokenizer.encode(text, add_special_tokens=False)))
        self.assertFalse(result["metadata"]["is_truncated"]
                         if len(tokens) <= 2 else True)

    def test_empty_input(self):
        text = ""
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=5)

        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["decoded_tokens"], [])
        self.assertEqual(result["texts"], [])
        self.assertEqual(result["metadata"]["total_tokens"], 0)
        self.assertFalse(result["metadata"]["is_truncated"])

    def test_single_sentence_within_limit(self):
        text = "Single sentence."
        result = merge_texts(text, self.tokenizer,
                             skip_special_tokens=True, max_length=5)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(result["texts"], [text])
        self.assertEqual(result["metadata"]["total_tokens"], len(tokens))
        self.assertFalse(result["metadata"]["is_truncated"])


if __name__ == "__main__":
    unittest.main()

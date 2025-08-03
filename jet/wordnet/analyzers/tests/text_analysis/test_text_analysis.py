import json
import unittest
import os
import random
from typing import TypedDict, Literal
from jet.utils.markdown import extract_json_block_content
from jet.wordnet.words import get_words
from textstat import textstat as ts
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger

# MLX setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated", "test_text_analysis")
MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"
MODEL: LLMModelType = "llama-3.2-1b-instruct-4bit"
SEED = 42
MLX_CLIENT = MLX(seed=SEED)


def generate_mlx_keyword_sentences(keywords, num_sentences=20) -> list[str]:
    """Generate sentences using MLX stream_generate based on keywords."""
    keywords_str = ", ".join(keywords)
    system_prompt = "You are a helpful AI assistant that follows instructions. You can generate coherent sentences with diversity given some provided keywords. You generate a JSON object of 'sentences' array surrounded by a JSON block (```json```). You output only a single JSON block without any additional text."
    prompt = f"Generate {num_sentences} short, coherent, valid sentences, each incorporating at least one of the following keywords: {keywords_str}. Ensure variety and avoid repetition."

    response = ""
    for stream_response in MLX_CLIENT.stream_chat(
        prompt,
        system_prompt=system_prompt,
        model=MODEL,
        temperature=0.3,
        log_dir=MLX_LOG_DIR,
        verbose=True,
    ):
        content = stream_response["choices"][0]["message"]["content"]
        response += content
        if stream_response["choices"][0]["finish_reason"]:
            break

    json_str = extract_json_block_content(response)
    result = json.loads(json_str)
    sentences = result["sentences"]

    return sentences


def extract_words_string(sentence: list[str], limit=10) -> str:
    word_count = 0
    results = []
    for line in sentence:
        if word_count >= limit:
            break
        # Assuming get_words splits a string into a list of words
        words = get_words(line)
        for word in words:
            if word_count < limit:
                results.append(word)
                word_count += 1
            else:
                break
    return " ".join(results)


class TestMTLDCalculator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        high_keywords = ["philosophy", "galaxy", "symphony",
                         "quantum", "culture", "mythology", "adventure"]
        sentences = generate_mlx_keyword_sentences(
            high_keywords, num_sentences=20)
        cls.short_text = extract_words_string(sentences, limit=1)
        cls.very_low_text = extract_words_string(sentences, limit=5)
        cls.low_text = extract_words_string(sentences, limit=10)
        cls.medium_text = extract_words_string(sentences, limit=20)
        cls.high_text = extract_words_string(sentences, limit=50)

        cls.very_low_stats = {
            "text_without_punctuation": ts.remove_punctuation(cls.very_low_text),
            "lexicon_count": ts.lexicon_count(cls.very_low_text)
        }
        cls.low_stats = {
            "text_without_punctuation": ts.remove_punctuation(cls.low_text),
            "lexicon_count": ts.lexicon_count(cls.low_text)
        }
        cls.medium_stats = {
            "text_without_punctuation": ts.remove_punctuation(cls.medium_text),
            "lexicon_count": ts.lexicon_count(cls.medium_text)
        }
        cls.high_stats = {
            "text_without_punctuation": ts.remove_punctuation(cls.high_text),
            "lexicon_count": ts.lexicon_count(cls.high_text)
        }
        cls.short_stats = {
            "text_without_punctuation": ts.remove_punctuation(cls.short_text),
            "lexicon_count": ts.lexicon_count(cls.short_text)
        }

    def test_calculate_mtld_short_text(self):
        score = calculate_mtld(self.__class__.short_stats)
        self.assertEqual(
            score, 0.0, f"Expected MTLD score 0.0 for short text, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(
            category, "very_low", f"Expected category 'very_low' for score 0.0, got {category}")

    def test_calculate_mtld_very_low(self):
        score = calculate_mtld(self.__class__.very_low_stats)
        self.assertLess(score, 40, f"Expected MTLD score < 40, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "very_low",
                         f"Expected category 'very_low', got {category}")

    def test_calculate_mtld_low(self):
        score = calculate_mtld(self.__class__.low_stats)
        self.assertTrue(40 <= score < 60,
                        f"Expected MTLD score in [40, 60), got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(
            category, "low", f"Expected category 'low', got {category}")

    def test_calculate_mtld_medium(self):
        score = calculate_mtld(self.__class__.medium_stats)
        self.assertTrue(60 <= score < 80,
                        f"Expected MTLD score in [60, 80), got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "medium",
                         f"Expected category 'medium', got {category}")

    def test_calculate_mtld_high(self):
        score = calculate_mtld(self.__class__.high_stats)
        self.assertGreaterEqual(
            score, 80, f"Expected MTLD score >= 80, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "high",
                         f"Expected category 'high', got {category}")

    def test_calculate_mtld_category_boundary_values(self):
        self.assertEqual(calculate_mtld_category(
            39.9), "very_low", "Expected 'very_low' for 39.9")
        self.assertEqual(calculate_mtld_category(
            40.0), "low", "Expected 'low' for 40.0")
        self.assertEqual(calculate_mtld_category(
            59.9), "low", "Expected 'low' for 59.9")
        self.assertEqual(calculate_mtld_category(
            60.0), "medium", "Expected 'medium' for 60.0")
        self.assertEqual(calculate_mtld_category(
            79.9), "medium", "Expected 'medium' for 79.9")
        self.assertEqual(calculate_mtld_category(
            80.0), "high", "Expected 'high' for 80.0")


if __name__ == '__main__':
    unittest.main()

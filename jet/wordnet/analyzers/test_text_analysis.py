import unittest
from typing import TypedDict, Literal
from textstat import textstat as ts
from jet.wordnet.analyzers.text_analysis import calculate_mtld, calculate_mtld_category


class TestMTLDCalculator(unittest.TestCase):
    def setUp(self):
        # Sample texts for each MTLD category
        self.very_low_text = "cat cat cat cat cat dog dog dog dog dog " * 10  # Highly repetitive
        self.low_text = ("the quick brown fox jumps over the lazy dog " * 5 +
                         "the cat runs fast and the dog sleeps")  # Moderate repetition
        self.medium_text = ("the quick brown fox jumps over the lazy dog. "
                            "a cat runs fast while a bird flies high. "
                            "trees grow tall and rivers flow deep. "
                            "children play games as adults work hard. "
                            "stars shine bright in the night sky.")  # Balanced diversity
        self.high_text = ("the swift cheetah sprints across vast savannas. "
                          "eagles soar majestically above rugged mountains. "
                          "coral reefs teem with vibrant marine biodiversity. "
                          "ancient civilizations crafted intricate artifacts. "
                          "quantum physicists unravel cosmic mysteries.")  # High diversity
        self.short_text = "cat dog bird"  # Fewer than 10 words

        # Prepare MLTDScores inputs
        self.very_low_stats = {
            "text_without_punctuation": ts.remove_punctuation(self.very_low_text),
            "lexicon_count": ts.lexicon_count(self.very_low_text)
        }
        self.low_stats = {
            "text_without_punctuation": ts.remove_punctuation(self.low_text),
            "lexicon_count": ts.lexicon_count(self.low_text)
        }
        self.medium_stats = {
            "text_without_punctuation": ts.remove_punctuation(self.medium_text),
            "lexicon_count": ts.lexicon_count(self.medium_text)
        }
        self.high_stats = {
            "text_without_punctuation": ts.remove_punctuation(self.high_text),
            "lexicon_count": ts.lexicon_count(self.high_text)
        }
        self.short_stats = {
            "text_without_punctuation": ts.remove_punctuation(self.short_text),
            "lexicon_count": ts.lexicon_count(self.short_text)
        }

    def test_calculate_mtld_very_low(self):
        score = calculate_mtld(self.very_low_stats["text_without_punctuation"])
        self.assertLess(score, 40, f"Expected MTLD score < 40, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "very_low",
                         f"Expected category 'very_low', got {category}")

    def test_calculate_mtld_low(self):
        score = calculate_mtld(self.low_stats["text_without_punctuation"])
        self.assertTrue(40 <= score < 60,
                        f"Expected MTLD score in [40, 60), got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(
            category, "low", f"Expected category 'low', got {category}")

    def test_calculate_mtld_medium(self):
        score = calculate_mtld(self.medium_stats["text_without_punctuation"])
        self.assertTrue(60 <= score < 80,
                        f"Expected MTLD score in [60, 80), got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "medium",
                         f"Expected category 'medium', got {category}")

    def test_calculate_mtld_high(self):
        score = calculate_mtld(self.high_stats["text_without_punctuation"])
        self.assertGreaterEqual(
            score, 80, f"Expected MTLD score >= 80, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(category, "high",
                         f"Expected category 'high', got {category}")

    def test_calculate_mtld_short_text(self):
        score = calculate_mtld(self.short_stats["text_without_punctuation"])
        self.assertEqual(
            score, 0.0, f"Expected MTLD score 0.0 for short text, got {score}")
        category = calculate_mtld_category(score)
        self.assertEqual(
            category, "very_low", f"Expected category 'very_low' for score 0.0, got {category}")

    def test_calculate_mtld_category_boundary_values(self):
        # Test boundary values for calculate_mtld_category
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

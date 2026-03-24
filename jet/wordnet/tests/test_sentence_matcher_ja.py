import unittest

from jet.wordnet.sentence_matcher_ja import FuzzyMatchResult, fuzzy_shortest_best_match


class TestFuzzyShortestBestMatch(unittest.TestCase):
    def test_perfect_match(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "潮ひ狩りはえ", "去年の初めての潮ひ狩りはえあうん楽しかったよ"
        )
        self.assertEqual(result["match"], "潮ひ狩りはえ")
        self.assertAlmostEqual(result["score"], 100.0, places=1)
        self.assertGreater(result["start"], -1)
        self.assertEqual(result["end"] - result["start"], len(result["match"]))

    def test_with_typo(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "去る初めての消ひ狩りはえ",
            "去る初めての潮ひ狩りはえあうん楽しかったよそうケン君は？",
        )
        self.assertEqual(result["match"], "去る初めての潮ひ狩りはえ")
        self.assertGreaterEqual(result["score"], 90.0)

    def test_shortest_on_score_tie(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "楽しかった", "楽しかったよそう楽しかったね"
        )
        self.assertEqual(result["match"], "楽しかった")
        self.assertLess(
            result["end"] - result["start"], len("楽しかったよそう楽しかったね")
        )

    def test_no_good_match(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "全く関係ない文字列です",
            "去る初めての潮ひ狩りはえあうん楽しかったよ",
            score_cutoff=30,
        )
        self.assertLess(result["score"], 30)
        self.assertEqual(result["start"], -1)

    def test_empty_inputs(self):
        empty: FuzzyMatchResult = fuzzy_shortest_best_match("", "abc")
        self.assertEqual(empty["match"], "")
        self.assertEqual(empty["score"], 0.0)
        self.assertEqual(empty["start"], -1)

        empty2: FuzzyMatchResult = fuzzy_shortest_best_match("abc", "")
        self.assertEqual(empty2["match"], "")
        self.assertEqual(empty2["score"], 0.0)

    def test_japanese_punctuation(self):
        result: FuzzyMatchResult = fuzzy_shortest_best_match(
            "ケン君は？", "楽しかったよそうケン君は？明日も行こうか"
        )
        self.assertEqual(result["match"], "ケン君は？")
        self.assertGreaterEqual(result["score"], 95.0)


if __name__ == "__main__":
    unittest.main()

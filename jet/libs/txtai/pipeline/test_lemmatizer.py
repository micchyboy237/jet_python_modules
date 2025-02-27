
import unittest
from jet.libs.txtai.pipeline.lemmatizer import lemmatize_text


class TestLemmatizeText(unittest.TestCase):
    def test_single_word(self):
        self.assertEqual(lemmatize_text("running"), ["running"])

    def test_sentences(self):
        self.assertEqual(lemmatize_text("The cats were running swiftly in the backyard."), [
                         "The", "cat", "were", "running", "swiftly", "in", "the", "backyard", "."])
        self.assertEqual(lemmatize_text("She has better ideas for the upcoming science fair."), [
                         "She", "ha", "better", "idea", "for", "the", "upcoming", "science", "fair", "."])
        self.assertEqual(lemmatize_text("Experienced programmers are coding sophisticated applications for artificial intelligence."), [
                         "Experienced", "programmer", "are", "coding", "sophisticated", "application", "for", "artificial", "intelligence", "."])
        self.assertEqual(lemmatize_text("The children are happily playing in the park despite the rain."), [
                         "The", "child", "are", "happily", "playing", "in", "the", "park", "despite", "the", "rain", "."])

    def test_plural_nouns(self):
        result = lemmatize_text("Dogs wolves mice")
        expected = [
            "Dogs",
            "wolf",
            "mouse"
        ]
        self.assertEqual(result, expected)

    def test_irregular_verbs(self):
        result = lemmatize_text("went gone going")
        expected = ["went", "gone", "going"]
        self.assertEqual(result, expected)

    def test_unicode_normalization(self):
        # En dash (–) should be converted to "-"
        result = lemmatize_text("Hello – World")
        expected = ["Hello", "-", "World"]
        self.assertEqual(result, expected)

    def test_unicode_characters(self):
        # Should convert accents
        result = lemmatize_text("Café naïve façade résumé")
        expected = ["Cafe", "naive", "facade", "resume"]
        self.assertEqual(result, expected)

    def test_with_software_dev_terms(self):
        sample = "React.js and JavaScript are widely used in modern web development to build interactive user interfaces."
        output = lemmatize_text(sample)
        expected = [
            "React.js",
            "and",
            "JavaScript",
            "are",
            "widely",
            "used",
            "in",
            "modern",
            "web",
            "development",
            "to",
            "build",
            "interactive",
            "user",
            "interface",
            "."
        ]

        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()

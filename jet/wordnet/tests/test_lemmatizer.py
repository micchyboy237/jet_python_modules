import unittest
from jet.wordnet.lemmatizer import lemmatize_text


class TestLemmatizeText(unittest.TestCase):
    def test_lemmatize_basic(self):
        sample = "The cats are running quickly."
        expected = "The cat are running quickly."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)

    def test_lemmatize_with_punctuation(self):
        sample = "I'll be going to John's house."
        expected = "I'll be going to John's house."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)

    def test_lemmatize_unicode(self):
        sample = "Café is a nice place. I’ve been there."
        expected = "Cafe is a nice place. I've been there."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)

    def test_lemmatize_numbers(self):
        sample = "There are 3 cars parked outside."
        expected = "There are 3 car parked outside."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)

    def test_lemmatize_contractions(self):
        sample = "You shouldn't be doing this."
        expected = "You shouldn't be doing this."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)

    def test_lemmatize_with_newlines(self):
        sample = "Hello world.\nThis is a test.\n\nLet's see if it works."
        expected = "Hello world.\nThis is a test.\n\nLet's see if it work."
        result = lemmatize_text(sample)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

import unittest
from similarity import (
    score_texts_similarity,
    get_similar_texts,
    differences,
    similars,
    compare_text_pairs,
    # has_close_match,
    # score_word_placement_similarity,
    # has_approximately_same_word_placement
)
# from instruction_generator.wordnet.SpellingCorrectorNorvig import SpellingCorrectorNorvig


class TestScoreTextsSimilarity(unittest.TestCase):
    def test_score_texts_similarity(self):
        text1 = '| | Learn Tagalog\n\nUseful Words and Phrases\nfor everyday use\n: (Part 1)'
        text2 = 'Learn Tagalog\n\nfor everyday use\n(Part 1)'
        score = score_texts_similarity(text1, text2)
        self.assertTrue(score >= 0.7, "Test failed for threshold 0.7")

        # Test with high similarity but not perfect match
        text3 = "ako ay mabuti\n\nIam fine"
        text4 = "ibuti\n\n1e"
        score = score_texts_similarity(text3, text4)
        self.assertFalse(score >= 0.7, "Test failed for threshold 0.7")

    def test_equal_texts(self):
        text1 = "I am going to the store"
        text2 = "I am going to the store"
        score = score_texts_similarity(text1, text2)
        self.assertTrue(score == 1, "Test failed for equal texts")


# class TestTextComparator(unittest.TestCase):
#     def test_text_comparator_1(self):
#         comparator = TextComparator()
#         text1 = '| | Learn Tagalog\n\nUseful Words and Phrases\nfor everyday use\n: (Part 1)'
#         text2 = 'Learn Tagalog\n\nfor everyday use\n(Part 1)'

#         self.assertTrue(comparator.contains_segments(
#             text1, text2), "Test failed for contains_segments")

#         # Test with high similarity but not perfect match
#         text3 = "ako ay mabuti\n\nIam fine"
#         text4 = "ibuti\n\n1e"
#         self.assertFalse(comparator.contains_segments(
#             text3, text4), "Test failed for contains_segments")

#     def test_text_comparator_2(self):
#         comparator = TextComparator()
#         text1 = 'Good morning\n\nMagandang umaga\n\nGood afternoon\n\nMagandang hapon'
#         text2 = 'Good afternoon\n\nMagandang hapon\n\nGood evening\n\nMagandang gabi'

#         self.assertFalse(comparator.contains_segments(
#             text1, text2), "Test failed for threshold 0.7")

#     # Test with empty texts

#     def test_text_comparator_enpty_text(self):
#         comparator = TextComparator()
#         text1 = 'Good morning'
#         text2 = ''

#         self.assertFalse(comparator.contains_segments(
#             text1, text2), "Test failed for empty texts")

#     def test_has_improved_spelling(self):
#         comparator = TextComparator()

#         # Test with improved spelling
#         base_text = "I am going to the stre."
#         updated_text = "I am going to the store"
#         self.assertTrue(comparator.has_improved_spelling(
#             updated_text, base_text), "Test failed for improved spelling")

#         # Test with same spelling
#         base_text = "I am going to the store"
#         updated_text = "I am going to the store"
#         self.assertFalse(comparator.has_improved_spelling(
#             updated_text, base_text), "Test failed for same spelling")

#         # Test with worse spelling
#         base_text = "I am going to the store"
#         updated_text = "I am going to the stre"
#         self.assertFalse(comparator.has_improved_spelling(
#             updated_text, base_text), "Test failed for worsened spelling")

#         # Test with no improvement
#         base_text = '| Learn Tagalog\n\nUseful Words and Phrases\nfor everyday use\n(Part 1)'
#         updated_text = '| Learn Tagalog\n\nUseful Words and Phrases\nfor everyday use\n\n| (Part 1)'
#         self.assertFalse(comparator.has_improved_spelling(
#             updated_text, base_text), "Test failed for no improvement")


class TestGetSimilarTexts(unittest.TestCase):
    def test_get_similar_texts(self):
        threshold = 0.7
        texts = [
            "October seven is the date of our vacation to Camarines Sur.",
            'We are going to Mindanao on January 8.',
            'October 7 is our vacation for Camarines Sur.',
        ]
        similar_texts = get_similar_texts(texts, threshold=threshold)
        # expected = [{'text1': 'October seven is the date of our vacation to Camarines Sur.We are going to Mindanao on January 8.', 'text2': 'October 7 is our vacation for Camarines Sur.', 'score': 0.5815602836879432}]
        # assert length
        self.assertEqual(len(similar_texts), 1,
                         "Test failed for get_similar_texts")
        # assert score
        self.assertTrue(similar_texts[0]['score'] >= threshold,
                        "Test failed for get_similar_texts")
        # assert texts
        self.assertEqual(similar_texts[0]['text1'], texts[0],
                         "Test failed for get_similar_texts")
        self.assertEqual(similar_texts[0]['text2'], texts[2],
                         "Test failed for get_similar_texts")


class TestDifferences(unittest.TestCase):

    def test_differences(self):
        texts = ["Hello world", "Hello beautiful world"]
        actual = differences(texts)
        expected = [{'text1': "Hello world",
                     'text2': "Hello beautiful world", 'differences': ['beautiful']}]
        self.assertEqual(actual, expected)

    def test_differences_full_sentences(self):
        texts = ["The quick brown fox jumps", "The quick brown cat leaps"]
        actual = differences(texts)
        expected = [{'text1': "The quick brown fox jumps", 'text2': "The quick brown cat leaps",
                     'differences': ['fox', 'jumps', 'cat', 'leaps']}]
        self.assertEqual(actual, expected)


class TestSimilars(unittest.TestCase):

    def test_similars(self):
        texts = ["Hello world", "Hello beautiful world"]
        actual = similars(texts)
        expected = [{'text1': "Hello world",
                     'text2': "Hello beautiful world", 'similars': ['Hello', 'world']}]
        self.assertEqual(actual, expected)

    def test_similars_full_sentences(self):
        texts = ["The quick brown fox jumps", "The quick brown cat leaps"]
        actual = similars(texts)
        expected = [{'text1': "The quick brown fox jumps",
                     'text2': "The quick brown cat leaps", 'similars': ['The', 'quick', 'brown']}]
        self.assertEqual(actual, expected)


class TestCompareTextPairs(unittest.TestCase):

    def test_similarities_and_differences(self):
        texts = ["Hello world", "Hello beautiful world"]
        expected = [{
            'text1': "Hello world",
            'text2': "Hello beautiful world",
            'similarities': ["Hello", "world"],
            'differences': ["beautiful"]
        }]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)

    def test_no_similarities(self):
        texts = ["Hello world", "Goodbye universe"]
        expected = [{
            'text1': "Hello world",
            'text2': "Goodbye universe",
            'similarities': [],
            'differences': ["Hello", "world", "Goodbye", "universe"]
        }]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)

    def test_full_sentences(self):
        texts = ["The quick brown fox jumps", "The quick brown cat leaps"]
        expected = [{
            'text1': "The quick brown fox jumps",
            'text2': "The quick brown cat leaps",
            'similarities': ["The", "quick", "brown"],
            'differences': ["fox", "jumps", "cat", "leaps"]
        }]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)

    def test_multiple_pairs(self):
        texts = ["Hello world", "Hello beautiful world", "Hello world"]
        expected = [
            {
                'text1': "Hello world",
                'text2': "Hello beautiful world",
                'similarities': ["Hello", "world"],
                'differences': ["beautiful"]
            },
            {
                'text1': "Hello beautiful world",
                'text2': "Hello world",
                'similarities': ["Hello", "world"],
                'differences': ["beautiful"]
            }
        ]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)

    def test_multiple_lines(self):
        texts = [
            "Hello world\nThis is a test\nEnd of message",
            "Hello universe\nThis is a test\nEnd of conversation"
        ]
        expected = [{
            'text1': "Hello world\nThis is a test\nEnd of message",
            'text2': "Hello universe\nThis is a test\nEnd of conversation",
            'similarities': ['Hello', 'This', 'is', 'a', 'test', 'End', 'of'],
            'differences': ['world', 'universe', 'message', 'conversation']
        }]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)

    def test_complex_multiple_lines(self):
        texts = [
            "First line\nSecond line\nThird line",
            "First line\nSecond modified line\nThird line",
            "First line\nTotally different line\nThird line"
        ]
        expected = [
            {
                'text1': "First line\nSecond line\nThird line",
                'text2': "First line\nSecond modified line\nThird line",
                'similarities': ['First', 'line', 'Second', 'line', 'Third', 'line'],
                'differences': ['modified']
            },
            {
                'text1': "First line\nSecond modified line\nThird line",
                'text2': "First line\nTotally different line\nThird line",
                'similarities': ['First', 'line', 'line', 'Third', 'line'],
                'differences': ['Second', 'modified', 'Totally', 'different']
            }
        ]
        actual = compare_text_pairs(texts)
        self.assertEqual(actual, expected)


# class TestHasCloseMatch(unittest.TestCase):
#     def test_has_close_match(self):
#         texts = [
#             "Ang aso mo’y kalbo.",
#             "Ang bait mo sa akin.",
#         ]
#         text = "Ang sakit ng ulo ko."
#         actual = has_close_match(text, texts)
#         expected = False
#         self.assertEqual(actual, expected)


# class TestScoreWordPlacementSimilarity(unittest.TestCase):
#     def test_equal(self):
#         word = "ako"
#         text1 = "Ako ay mabuti"
#         text2 = "Ako si Juan"
#         actual = score_word_placement_similarity(word, text1, text2)
#         self.assertEqual(actual, 1.0)

#     def test_almost_equal(self):
#         word = "ako"
#         text1 = "Ako ay mabuti"
#         text2 = "Pumunta ako sa tindahan"
#         actual = score_word_placement_similarity(word, text1, text2)
#         # In text1, "ako" is at the start (0/3), in text2 it's one word later (1/4)
#         expected = 1.0 - abs((0/3) - (1/4))
#         self.assertEqual(actual, expected)

#     def test_different(self):
#         word = "ako"
#         text1 = "Ako ay mabuti"
#         text2 = "Siya ay mabuti, hindi ako"
#         actual = score_word_placement_similarity(word, text1, text2)
#         # In text1, "ako" is at the start (0/3), in text2 it's at the end (5/5)
#         expected = 1.0 - abs((0/3) - (4/5))
#         self.assertEqual(actual, expected)

#     def test_word_not_in_texts(self):
#         word = "ako"
#         text1 = "Ako ay mabuti"
#         text2 = "Siya ay mabuti"
#         actual = score_word_placement_similarity(word, text1, text2)
#         self.assertEqual(actual, 0.0)


# class TestHasApproximatelySameWordPlacement(unittest.TestCase):
#     def test_same_position_small_texts(self):
#         word = "aba"
#         text = "Aba, ano na ang ginawa mo ngayong araw?"
#         texts = ["Aba, sa wakas may nahuli rin siya!"]
#         actual = has_approximately_same_word_placement(word, text, texts)
#         self.assertTrue(actual)

#     def test_same_position_different_n_value_present(self):
#         word = "aba"
#         n = 2
#         text = "Aba, ano na ang ginawa mo ngayong araw?"
#         texts = ["Aba, ano ba ang pinagkaiba ngayon?"]
#         actual = has_approximately_same_word_placement(word, text, texts, n=n)
#         self.assertTrue(actual)

#     def test_same_position_different_n_value_not_present(self):
#         word = "aba"
#         n = 2
#         text = "Aba, ano na ang ginawa mo ngayong araw?"
#         texts = ["Aba, sa wakas may nahuli rin siya!"]
#         actual = has_approximately_same_word_placement(word, text, texts, n=n)
#         self.assertFalse(actual)

#     def test_same_position_large_texts(self):
#         word = "test"
#         text = "This is a test sentence with more words"
#         texts = ["In this sentence, the word test appears later"]
#         # Even though 'test' is not in the exact same position, the function should return True
#         # due to the leniency in larger texts.
#         actual = has_approximately_same_word_placement(word, text, texts)
#         self.assertFalse(actual)

#     def test_different_position(self):
#         word = "ako"
#         text = "Ako ay mabuti"
#         texts = ["Siya ay mabuti, hindi ako"]
#         # Here 'ako' is in a different position, so it should return False.
#         actual = has_approximately_same_word_placement(word, text, texts)
#         self.assertFalse(actual)

#     def test_word_not_present(self):
#         word = "ako"
#         text = "Ako ay mabuti"
#         texts = ["Walang salitang ito"]
#         # Since the word is not present in the other text, it should return False.
#         actual = has_approximately_same_word_placement(word, text, texts)
#         self.assertFalse(actual)

#     def test_multiple_texts(self):
#         word = "ako"
#         text = "Ako ay mabuti"
#         texts = ["Ako rin", "Hindi ako", "Siya ay ako"]
#         # The word 'ako' has a similar position in at least one of the texts,
#         # so the function should return True.
#         actual = has_approximately_same_word_placement(word, text, texts)
#         self.assertTrue(actual)


if __name__ == "__main__":
    unittest.main()

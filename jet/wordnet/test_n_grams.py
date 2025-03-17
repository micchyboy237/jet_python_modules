from collections import Counter
from jet.wordnet.words import count_words
from jet.wordnet.n_grams import (
    calculate_n_gram_diversity,
    count_ngrams,
    extract_ngrams,
    filter_texts_by_multi_ngram_count,
    get_most_common_ngrams,
    get_ngram_weight,
    get_ngrams_by_range,
    get_specific_ngram_count,
    get_total_counts_of_ngrams,
    get_total_unique_ngrams,
    group_sentences_by_ngram,
    filter_and_sort_sentences_by_ngrams,
    n_gram_frequency,
    separate_ngram_lines,
    sort_sentences,
    nwise,
)
from jet.wordnet.similarity import are_texts_similar, filter_similar_texts
import unittest


class TestSeparateNgramLines(unittest.TestCase):

    def test_single_string_input(self):
        # Test with a single string input
        sample = "This is a test sentence. Let's split it."
        expected = ["This is a test sentence.", "Let's split it."]

        result = separate_ngram_lines(sample)

        self.assertEqual(result, expected)

    def test_list_of_strings_input(self):
        # Test with a list of strings input
        sample = [
            "This is the first sentence.",
            "And here is the second one, with commas."
        ]
        expected = [
            "This is the first sentence.",
            "And here is the second one",
            "with commas."
        ]

        result = separate_ngram_lines(sample)

        self.assertEqual(result, expected)

    def test_with_different_punctuation(self):
        # Test with different punctuation
        sample = "Test/with:multiple,punctuations"
        expected = ['Test', 'with', 'multiple', 'punctuations']

        result = separate_ngram_lines(sample, [',', '/', ':'])

        self.assertEqual(result, expected)

    def test_repeated_sentences(self):
        # Test when the input has repeated sentences
        sample = "This is a test. This is a test."
        expected = ["This is a test."]

        result = separate_ngram_lines(sample)

        self.assertEqual(result, expected)

    def test_empty_string(self):
        # Test with an empty string
        sample = ""
        expected = []

        result = separate_ngram_lines(sample)

        self.assertEqual(result, expected)

    def test_edge_case_special_characters(self):
        # Test with special characters in the input
        sample = "Hello! #Testing$%&"
        expected = ['Hello', 'Testing']

        result = separate_ngram_lines(sample, ['!', '#', '$', '%', '&'])

        self.assertEqual(result, expected)


class TestExtractNgrams(unittest.TestCase):

    def test_single_word_ngram(self):
        texts = "Hello"
        result = extract_ngrams(texts, min_words=1, max_words=1)
        self.assertEqual(result, ["Hello"])

    def test_multiple_word_ngram(self):
        texts = "Hello world"
        result = extract_ngrams(texts, min_words=1, max_words=2)
        expected = ["Hello", "world", "Hello world"]
        self.assertCountEqual(result, expected)

    def test_empty_text(self):
        texts = ""
        result = extract_ngrams(texts, min_words=1, max_words=1)
        self.assertEqual(result, [])

    def test_list_input(self):
        texts = ["Hello", "world"]
        result = extract_ngrams(texts, min_words=1, max_words=2)
        expected = ["Hello", "world"]
        self.assertCountEqual(result, expected)

    def test_custom_ngram_size(self):
        texts = "I am learning Python"
        result = extract_ngrams(texts, min_words=2, max_words=3)
        expected = ['I am', 'am learning', 'learning Python',
                    'I am learning', 'am learning Python']
        self.assertCountEqual(result, expected)


class TestCountNgrams(unittest.TestCase):

    def test_count_single_word_ngram(self):
        texts = "Hello world Hello"
        result = count_ngrams(texts, min_words=1, min_count=1)
        expected = {"Hello": 2, "world": 1}
        self.assertEqual(result, expected)

    def test_count_multiple_word_ngram(self):
        texts = "I am learning Python, I am learning"
        result = count_ngrams(texts, min_words=2, min_count=1)
        expected = {"I am": 2, "am learning": 2, "learning Python": 1}
        self.assertEqual(result, expected)

    def test_ngram_with_min_count(self):
        texts = "apple apple banana banana apple"
        result = count_ngrams(texts, min_words=1, min_count=2)
        expected = {"apple": 3, "banana": 2}
        self.assertEqual(result, expected)

    def test_empty_text(self):
        texts = ""
        result = count_ngrams(texts, min_words=1, min_count=1)
        self.assertEqual(result, {})

    def test_list_input(self):
        texts = ["I am learning", "Python is fun"]
        result = count_ngrams(texts, min_words=2, min_count=1)
        expected = {"I am": 1, "am learning": 1, "Python is": 1, "is fun": 1}
        self.assertEqual(result, expected)


class TestGetMostCommonNgrams(unittest.TestCase):

    def test_basic_functionality(self):
        texts = "apple banana apple orange apple banana"
        result = get_most_common_ngrams(texts, min_count=2, max_words=2)
        expected = {"apple": 3, "banana": 2, "apple banana": 2}
        self.assertEqual(result, expected)

    def test_stop_words_removal(self):
        texts = "the quick brown fox jumps over the lazy dog"
        result = get_most_common_ngrams(texts, min_count=1, max_words=2)
        expected = {"quick": 1, "brown": 1, "fox": 1, "jumps": 1, "lazy": 1,
                    "dog": 1, "quick brown": 1, "brown fox": 1, "fox jumps": 1, "lazy dog": 1}
        self.assertEqual(result, expected)

    def test_empty_text(self):
        texts = ""
        result = get_most_common_ngrams(texts, min_count=2, max_words=2)
        self.assertEqual(result, {})

    def test_list_input(self):
        texts = ["I am learning", "Python is fun"]
        result = get_most_common_ngrams(texts, min_count=1, max_words=2)
        expected = {"learning": 1, "python": 1, "fun": 1}
        self.assertEqual(result, expected)


class TestGetMostCommonNgrams2(unittest.TestCase):

    def test_multiple_sentences_with_min_count(self):
        # Test with multiple sentences and min_count argument
        sample_texts = [
            "The sun has risen.",
            "The night is dark.",
            "The sun rose and the night fell."
        ]
        expected_output = {
            "sun": 2,
            "risen": 1,
            "sun has risen": 1,
            "night": 2,
            "dark": 1,
            "night is dark": 1,
            "rose": 1,
            "fell": 1,
            "sun rose": 1,
            "night fell": 1,
            "rose and the night": 1,
            "sun rose and the night": 1,
            "rose and the night fell": 1
        }
        result = get_most_common_ngrams(sample_texts, min_count=1)
        self.assertEqual(result, expected_output)

    def test_with_n_range(self):
        # Test with n range (2, 3)
        sample_texts = [
            "The sun has risen.",
            "The night is dark and the sun rose.",
            "The sun rose and the night fell."
        ]
        expected_output = {"sun rose": 2}
        result = get_most_common_ngrams(
            sample_texts, min_count=2, min_words=2, max_words=3)
        self.assertEqual(result, expected_output)

    def test_large_text_input_with_n_range(self):
        # Test with a long paragraph and n range (2, 3)
        sample_text = ("The sun has risen and the night has passed. "
                       "The people worked through the day. "
                       "The sun rose and the night came. "
                       "Stars shine brightly at night.")
        expected_output = {
            "night": 3,
            "sun": 2
        }
        result = get_most_common_ngrams(sample_text, min_count=2, max_words=3)
        self.assertEqual(result, expected_output)

    def test_no_common_ngrams(self):
        # Test where there are no common n-grams
        sample_text = "Hello world."
        expected_output = {}
        result = get_most_common_ngrams(sample_text, min_count=2)
        self.assertEqual(result, expected_output)

    def test_empty_input(self):
        # Test with an empty input
        sample_text = ""
        expected_output = {}
        result = get_most_common_ngrams(sample_text)
        self.assertEqual(result, expected_output)


class TestSortSentences(unittest.TestCase):

    def test_even_distribution(self):
        sentences = [
            "Ilarawan ang istruktura ni boi",
            "Ilarawan ang istruktura sa buhok",
            "Dahil ang istruktura na mahalaga",
            "Magbigay ng tatlong tip",
            "Paano natin mababawasan?",
            "Kailangan mong gumawa ng isang mahirap na desisyon.",
            "Kilalanin ang gumawa ng desisyon na iyon.",
            "Ipaliwanag kung bakit",
            "Sumulat ng isang maikling kuwento",
        ]
        sorted_sentences = sort_sentences(sentences, 2)
        # Expecting that sentences starting with the same n-grams are not adjacent
        self.assertNotEqual(sorted_sentences[0].split()[
                            0], sorted_sentences[1].split()[0])
        self.assertNotEqual(sorted_sentences[2].split()[
                            0], sorted_sentences[3].split()[0])

    def test_large_dataset(self):
        sentences = ["Sentence " + str(i) for i in range(100)]
        sorted_sentences = sort_sentences(sentences, 2)
        # Check if the sorted list is the same length as the input
        self.assertEqual(len(sorted_sentences), 100)
        # Check for null values in sorted_sentences
        self.assertFalse(
            any(sentence is None for sentence in sorted_sentences))

    def test_small_dataset(self):
        sentences = [
            "Paraphrase this sentence.",
            "Another sentence."
        ]
        sorted_sentences = sort_sentences(sentences, 2)
        # Check if the sorted list is the same length as the input
        self.assertEqual(len(sorted_sentences), 2)


class TestGroupAndFilterByNgram(unittest.TestCase):

    def test_is_start_ngrams(self):
        sentences = [
            "How are you today?",
            "How are you doing?",
            "How are you doing today?",
            "Thank you for asking.",
            "Thank you again",
            "Thank you"
        ]
        n = 2
        top_n = 2
        result = group_sentences_by_ngram(sentences, n, top_n, True)
        expected_grouping = {
            'How are': ['How are you today?', 'How are you doing?'],
            'Thank you': ['Thank you', 'Thank you again']
        }
        self.assertDictEqual(result, expected_grouping,
                             "Sentences are not grouped correctly.")

    # def test_offset_n(self):
    #     sentences = [
    #         "The quick brown fox jumps over the lazy dog",
    #         "Quick as a fox, sharp as an eagle",
    #         "The lazy dog sleeps soundly",
    #         "A quick brown dog leaps over a lazy fox"
    #     ]
    #     n = 2
    #     top_n = 2
    #     result = group_sentences_by_ngram(sentences, n, top_n, False)
    #     expected_grouping = {
    #         'quick brown': ["The quick brown fox jumps over the lazy dog", "A quick brown dog leaps over a lazy fox"],
    #         'as a': ["Quick as a fox, sharp as an eagle", "A quick brown dog leaps over a lazy fox"]
    #     }
    #     self.assertDictEqual(result, expected_grouping,
    #                          "Sentences are not grouped correctly for non-start n-grams.")


class TestSentenceProcessing(unittest.TestCase):

    def test_group_and_limit_sentences(self):
        sentences = [
            "Paraphrase the following sentence.",
            "Paraphrase a different sentence.",
            "Another example sentence.",
            "Yet another example sentence."
        ]
        sorted_sentences = filter_and_sort_sentences_by_ngrams(
            sentences, 1, 1, True)
        # Expecting only one sentence per unique starting n-gram
        self.assertEqual(len(sorted_sentences), 3)

    def test_spread_sentences(self):
        sentences = [
            "Combine these sentences.",
            "Combine those sentences.",
            "An example sentence.",
            "Another example sentence."
        ]
        sorted_sentences = filter_and_sort_sentences_by_ngrams(
            sentences, 2, 2, True)
        # Expecting the "Combine" sentences to be spread out
        self.assertNotEqual(sorted_sentences[0].split()[
                            0], sorted_sentences[1].split()[0])

    def test_filter_similar_texts(self):
        sentences = [
            "This is a sentence.",
            "This is a sentence!",
            "This is another sentence.",
            "A completely different sentence."
        ]
        filtered_sentences = filter_similar_texts(sentences)
        expected_sentences = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        # Expecting the very similar sentences to be filtered out
        self.assertEqual(filtered_sentences, expected_sentences)

    def test_filter_similar_texts_identical(self):
        sentences = ["Hello world", "Hello world", "Hello world"]
        filtered = filter_similar_texts(sentences)
        self.assertEqual(len(filtered), 1)

    def test_filter_similar_texts_different(self):
        sentences = ["Hello world", "Goodbye world", "How are you"]
        filtered = filter_similar_texts(sentences)
        self.assertEqual(len(filtered), len(sentences))

    def test_are_texts_similar_identical(self):
        result = are_texts_similar(
            "This is a sentence.", "This is another sentence.")
        self.assertTrue(result)

    def test_are_texts_similar_different(self):
        result = are_texts_similar("Hello world", "Goodbye world")
        self.assertFalse(result)


class TestNwise(unittest.TestCase):

    def test_single_element(self):
        """Test with min_words=1, should return individual elements."""
        data = [1, 2, 3, 4]
        expected = [(1,), (2,), (3,), (4,)]
        result = list(nwise(data, 1))
        self.assertEqual(result, expected)

    def test_pairwise(self):
        """Test with min_words=2, should return pairs of elements."""
        data = 'abcd'
        expected = [('a', 'b'), ('b', 'c'), ('c', 'd')]
        result = list(nwise(data, 2))
        self.assertEqual(result, expected)

    def test_unigrams_with_sentence(self):
        """Test with min_words=1 on a sentence, should return individual words."""
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The',), ('quick',), ('brown',), ('fox',), ('jumps',
                                                         ), ('over',), ('the',), ('lazy',), ('dog',)
        ]
        result = list(nwise(sentence, 1))
        self.assertEqual(result, expected)

    def test_triplets_with_sentence(self):
        """Test with min_words=3 on a sentence, should return triplets of words."""
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The', 'quick', 'brown'),
            ('quick', 'brown', 'fox'),
            ('brown', 'fox', 'jumps'),
            ('fox', 'jumps', 'over'),
            ('jumps', 'over', 'the'),
            ('over', 'the', 'lazy'),
            ('the', 'lazy', 'dog')
        ]
        result = list(nwise(sentence, 3))
        self.assertEqual(result, expected)

    def test_empty_iterable(self):
        """Test with an empty iterable, should return an empty iterable."""
        data = []
        expected = []
        result = list(nwise(data, 2))
        self.assertEqual(result, expected)

    def test_large_n(self):
        """Test with n larger than the length of the iterable, should return an empty iterable."""
        data = [1, 2, 3]
        expected = []
        result = list(nwise(data, 5))
        self.assertEqual(result, expected)


# class TestNgramFunctions(unittest.TestCase):

#     def test_n_gram_frequency(self):
#         sentence = "I love coding"
#         expected_result = {('I', 'love'): 1, ('love', 'coding'): 1}
#         result = n_gram_frequency(sentence.split(), 2)
#         self.assertEqual(result, expected_result)

#     def test_calculate_n_gram_diversity(self):
#         freq = Counter({('I', 'love'): 3, ('love', 'coding'): 2})
#         expected_result = 2  # There are two unique n-grams
#         result = calculate_n_gram_diversity(freq)
#         self.assertEqual(result, expected_result)

#     def test_get_ngram_weight(self):
#         all_ngrams = Counter({('I', 'love'): 3, ('love', 'coding'): 2})
#         sentence_ngrams = {('I', 'love')}
#         previous_ngrams = {('love', 'coding')}
#         expected_result = 1 / 3 + 1  # 1/ngram frequency + penalty
#         result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
#         self.assertEqual(result, expected_result)

#     def test_get_total_unique_ngrams(self):
#         ngram_counter = Counter({('I', 'love'): 3, ('love', 'coding'): 2})
#         expected_result = 2  # Two unique n-grams
#         result = get_total_unique_ngrams(ngram_counter)
#         self.assertEqual(result, expected_result)

#     def test_get_total_counts_of_ngrams(self):
#         ngram_counter = Counter({('I', 'love'): 3, ('love', 'coding'): 2})
#         expected_result = 5  # Total count of all n-grams
#         result = get_total_counts_of_ngrams(ngram_counter)
#         self.assertEqual(result, expected_result)

#     def test_get_specific_ngram_count(self):
#         ngram_counter = Counter({('I', 'love'): 3, ('love', 'coding'): 2})
#         specific_ngram = ('I', 'love')
#         expected_result = 3
#         result = get_specific_ngram_count(ngram_counter, specific_ngram)
#         self.assertEqual(result, expected_result)

#     def test_get_ngrams_by_range(self):
#         texts = ["I love coding", "I love learning", "Coding is fun"]
#         expected_result = [
#             {
#                 "ngram": "I love",
#                 "count": 2
#             }
#         ]
#         result = get_ngrams_by_range(texts, min_words=(2,), count=2, show_count=True)
#         self.assertEqual(result, expected_result)

#     def test_filter_texts_by_multi_ngram_count(self):
#         texts = [
#             "I love coding",
#             "I love learning",
#             "Coding is fun"
#         ]
#         n = 2
#         count = (2,)
#         expected_result = ["I love coding"]
#         result = filter_texts_by_multi_ngram_count(texts, n, count)
#         self.assertEqual(result, expected_result)

#     def test_nwise(self):
#         iterable = ['A', 'B', 'C', 'D']
#         expected_result = [('A', 'B'), ('B', 'C'), ('C', 'D')]
#         result = list(nwise(iterable, 2))
#         self.assertEqual(result, expected_result)

# class TestFilterSentencesByPosTags(unittest.TestCase):
#     def test_filter_sentences_by_pos_tags(self):
#         sentences = [
#             "How are you today?",
#             "How are you doing?",
#             "How are you doing today?",
#             "Thank you for asking.",
#             "Thank you again",
#             "Thank you"
#         ]
#         pos_tags = ["PRON", "VERB", "ADV"]
#         expected = [
#             "How are you today?",
#             "How are you doing?",
#             "How are you doing today?",
#         ]
#         result = filter_sentences_by_pos_tags(sentences, pos_tags)
#         self.assertEqual(result, expected,
#                          "Sentences are not filtered correctly.")
if __name__ == '__main__':
    unittest.main()

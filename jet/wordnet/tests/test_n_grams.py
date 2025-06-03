from jet.wordnet.stopwords import StopWords
from jet.wordnet.pos_tagger import POSTagger, POSItem
from collections import Counter
from jet.wordnet.words import count_words
from jet.wordnet.n_grams import (
    calculate_n_gram_diversity,
    count_ngrams,
    extract_ngrams,
    filter_texts_by_multi_ngram_count,
    get_common_texts,
    get_most_common_ngrams,
    get_ngram_weight,
    get_ngrams,
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
        sample = "This is a test sentence. Let's split it."
        expected = ["This is a test sentence.", "Let's split it."]
        result = separate_ngram_lines(sample)
        self.assertEqual(result, expected)

    def test_list_of_strings_input(self):
        sample = [
            "This is the first sentence.",
            "And here is the second one, with commas."
        ]
        expected = [
            "And here is the second one",
            "This is the first sentence.",
            "with commas."
        ]
        result = separate_ngram_lines(sample)
        self.assertEqual(result, expected)

    def test_with_different_punctuation(self):
        sample = "Test/with:multiple,punctuations"
        expected = ['Test', 'multiple', 'punctuations', 'with']
        result = separate_ngram_lines(sample, [',', '/', ':'])
        self.assertEqual(result, expected)

    def test_repeated_sentences(self):
        sample = "This is a test. This is a test."
        expected = ["This is a test."]
        result = separate_ngram_lines(sample)
        self.assertEqual(result, expected)

    def test_empty_string(self):
        sample = ""
        expected = []
        result = separate_ngram_lines(sample)
        self.assertEqual(result, expected)

    def test_edge_case_special_characters(self):
        sample = "Hello!Testing"
        expected = ['Hello', 'Testing']
        result = separate_ngram_lines(sample, ['!', '\n'])
        self.assertEqual(result, expected)


class TestExtractNgrams(unittest.TestCase):
    def test_single_word_ngram(self):
        texts = "Hello"
        result = extract_ngrams(texts, min_words=1, max_words=1)
        expected = ["Hello"]
        self.assertEqual(result, expected)

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
        expected = {"Hello": 2, "world": 1, "Hello world": 1,
                    "world Hello": 1, "Hello world Hello": 1}
        self.assertEqual(result, expected)

    def test_count_multiple_word_ngram(self):
        texts = "I am learning Python, I am learning"
        result = count_ngrams(texts, min_words=2, min_count=1)
        expected = {
            "I am": 2,
            "am learning": 2,
            "learning Python": 1,
            "I am learning": 2,
            "am learning Python": 1,
            "I am learning Python": 1
        }
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
        expected = {"I am": 1, "am learning": 1, "I am learning": 1,
                    "Python is": 1, "is fun": 1, "Python is fun": 1}
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
        texts = ["I am learning Python", "I am having fun learning Python"]
        result = get_most_common_ngrams(texts, min_count=1, max_words=2)
        expected = {"learning": 2, "python": 2,
                    "learning python": 2, "fun": 1, "fun learning": 1}
        self.assertEqual(result, expected)


class TestGetMostCommonNgrams2(unittest.TestCase):
    def test_multiple_sentences_with_min_count(self):
        sample_texts = [
            "The sun has risen.",
            "The night is dark.",
            "The sun rose and the night fell."
        ]
        expected_output = {"sun": 2, "risen": 1, "sun has risen": 1, "night": 2, "dark": 1, "night is dark": 1, "rose": 1, "fell": 1, "sun rose": 1,
                           "night fell": 1, "rose and the night": 1, "sun rose and the night": 1, "rose and the night fell": 1, "sun rose and the night fell": 1}
        result = get_most_common_ngrams(sample_texts, min_count=1)
        self.assertEqual(result, expected_output)

    def test_with_n_range(self):
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
        sample_text = "Hello world."
        expected_output = {}
        result = get_most_common_ngrams(sample_text, min_count=2)
        self.assertEqual(result, expected_output)

    def test_empty_input(self):
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
        self.assertNotEqual(sorted_sentences[0].split()[
                            0], sorted_sentences[1].split()[0])
        self.assertNotEqual(sorted_sentences[2].split()[
                            0], sorted_sentences[3].split()[0])

    def test_large_dataset(self):
        sentences = ["Sentence " + str(i) for i in range(100)]
        sorted_sentences = sort_sentences(sentences, 2)
        self.assertEqual(len(sorted_sentences), 100)
        self.assertFalse(
            any(sentence is None for sentence in sorted_sentences))

    def test_small_dataset(self):
        sentences = [
            "Paraphrase this sentence.",
            "Another sentence."
        ]
        sorted_sentences = sort_sentences(sentences, 2)
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


class TestGetNgrams(unittest.TestCase):
    def test_single_word(self):
        sample = "hello"
        expected = ["hello"]
        result = get_ngrams(sample)
        self.assertEqual(result, expected)

    def test_multiple_words(self):
        sample = "hello world"
        expected = ["hello", "world", "hello world"]
        result = get_ngrams(sample, min_words=1, max_words=2)
        self.assertCountEqual(result, expected)

    def test_min_words(self):
        sample = "hello world example"
        expected = ["hello world", "world example", "hello world example"]
        result = get_ngrams(sample, min_words=2)
        self.assertCountEqual(result, expected)

    def test_min_count(self):
        sample = ["hello world", "hello world", "example"]
        expected = ["hello world"]
        result = get_ngrams(sample, min_words=2, min_count=2)
        self.assertEqual(result, expected)

    def test_max_words(self):
        sample = "this is a test"
        expected = ["this", "is", "a", "test", "this is", "is a", "a test"]
        result = get_ngrams(sample, max_words=2)
        self.assertCountEqual(result, expected)

    def test_empty_input(self):
        sample = ""
        expected = []
        result = get_ngrams(sample)
        self.assertEqual(result, expected)


class TestGetCommonTexts(unittest.TestCase):
    def setUp(self):
        self.tagger = POSTagger()
        self.stopwords = StopWords()

    def test_single_text(self):
        texts = ["The quick brown fox jumps"]
        expected = ['quick brown fox jumps']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "VERB", "ADJ", "PROPN"], min_words=1, max_words=4)
        self.assertEqual(result, expected)

    def test_multiple_texts(self):
        texts = [
            "The quick brown fox jumps",
            "Quick brown fox runs fast"
        ]
        expected = ['quick brown fox']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "VERB", "ADJ", "PROPN"], min_words=1, max_words=3)
        self.assertEqual(result, expected)

    def test_empty_input(self):
        texts = []
        expected = []
        result = get_common_texts(texts)
        self.assertEqual(result, expected)

    def test_stopwords_filtering(self):
        texts = ["The quick fox is running"]
        expected = ['quick fox running']
        result = get_common_texts(texts, min_words=1, max_words=3)
        self.assertEqual(result, expected)

    def test_different_pos_tags(self):
        texts = ["The quick brown fox"]
        expected = ['quick brown fox']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "ADJ"], min_words=1, max_words=3)
        self.assertEqual(result, expected)

    def test_no_matching_pos(self):
        texts = ["The quick brown fox"]
        expected = []
        result = get_common_texts(texts, includes_pos=["VERB"])
        self.assertEqual(result, expected)

    def test_case_insensitivity(self):
        texts = ["Quick Brown Fox", "quick brown fox"]
        expected = ['quick brown fox']
        result = get_common_texts(texts, min_words=1, max_words=3)
        self.assertEqual(result, expected)


class TestGroupSentencesByNgram(unittest.TestCase):
    def setUp(self):
        self.sentences = [
            "the quick brown fox",
            "the quick brown dog",
            "a slow green turtle",
            "the fast blue bird",
            "a quick brown cat"
        ]

    def test_start_ngrams_single_word(self):
        result = group_sentences_by_ngram(
            self.sentences, min_words=1, top_n=2, is_start_ngrams=True)
        expected = {
            'the': ['the quick brown fox', 'the quick brown dog'],
            'a': ['a slow green turtle', 'a quick brown cat']
        }
        self.assertEqual(result, expected)

    def test_start_ngrams_two_words(self):
        result = group_sentences_by_ngram(
            self.sentences, min_words=2, top_n=2, is_start_ngrams=True)
        expected = {
            'the quick': ['the quick brown fox', 'the quick brown dog'],
            'a slow': ['a slow green turtle'],
            'the fast': ['the fast blue bird'],
            'a quick': ['a quick brown cat']
        }
        self.assertEqual(result, expected)

    def test_non_start_ngrams(self):
        result = group_sentences_by_ngram(
            self.sentences, min_words=2, top_n=2, is_start_ngrams=False)
        self.assertTrue(isinstance(result, dict))
        for ngram, sentences in result.items():
            self.assertTrue(isinstance(ngram, str))
            self.assertTrue(isinstance(sentences, list))
            self.assertLessEqual(len(sentences), 2)
            for sentence in sentences:
                self.assertIn(sentence, self.sentences)

    def test_top_n_limit(self):
        sentences = [
            "the quick brown fox",
            "the quick brown dog",
            "the quick brown cat",
            "the quick brown bird"
        ]
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        expected = {
            'the quick': ['the quick brown fox', 'the quick brown dog']
        }
        self.assertEqual(result, expected)

    def test_empty_sentences(self):
        result = group_sentences_by_ngram(
            [], min_words=2, top_n=2, is_start_ngrams=True)
        self.assertEqual(result, {})

    def test_sentences_with_no_ngrams(self):
        sentences = ["the", "a"]
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        self.assertEqual(result, {})

    def test_sorting_by_word_count_and_index(self):
        sentences = [
            "the quick brown fox jumps",
            "the quick brown dog",
            "the quick brown cat runs"
        ]
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        expected = {
            'the quick': [
                'the quick brown dog',
                'the quick brown fox jumps'
            ]
        }
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

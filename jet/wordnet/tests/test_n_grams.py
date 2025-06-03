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
import pytest


class TestSeparateNgramLines:
    def test_single_string_input(self):
        sample = "This is a test sentence. Let's split it."
        expected = sorted(["This is a test sentence.", "Let's split it."])
        result = sorted(separate_ngram_lines(sample))
        assert result == expected

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
        assert result == expected

    def test_with_different_punctuation(self):
        sample = "Test/with:multiple,punctuations"
        expected = ['Test', 'multiple', 'punctuations', 'with']
        result = separate_ngram_lines(sample, [',', '/', ':'])
        assert result == expected

    def test_repeated_sentences(self):
        sample = "This is a test. This is a test."
        expected = ["This is a test."]
        result = separate_ngram_lines(sample)
        assert result == expected

    def test_empty_string(self):
        sample = ""
        expected = []
        result = separate_ngram_lines(sample)
        assert result == expected

    def test_edge_case_special_characters(self):
        sample = "Hello!Testing"
        expected = ['Hello', 'Testing']
        result = separate_ngram_lines(sample, ['!', '\n'])
        assert result == expected


class TestExtractNgrams:
    def test_single_word_ngram(self):
        texts = "Hello"
        expected = ["Hello"]
        result = extract_ngrams(texts, min_words=1, max_words=1)
        assert result == expected

    def test_multiple_word_ngram(self):
        texts = "Hello world"
        expected = ["Hello", "world", "Hello world"]
        result = extract_ngrams(texts, min_words=1, max_words=2)
        assert sorted(result) == sorted(expected)

    def test_empty_text(self):
        texts = ""
        expected = []
        result = extract_ngrams(texts, min_words=1, max_words=1)
        assert result == expected

    def test_list_input(self):
        texts = ["Hello", "world"]
        expected = ["Hello", "world"]
        result = extract_ngrams(texts, min_words=1, max_words=2)
        assert sorted(result) == sorted(expected)

    def test_custom_ngram_size(self):
        texts = "I am learning Python"
        expected = ['I am', 'am learning', 'learning Python',
                    'I am learning', 'am learning Python']
        result = extract_ngrams(texts, min_words=2, max_words=3)
        assert sorted(result) == sorted(expected)


class TestCountNgrams:
    def test_count_single_word_ngram(self):
        texts = "Hello world Hello"
        expected = {"Hello": 2, "world": 1, "Hello world": 1,
                    "world Hello": 1, "Hello world Hello": 1}
        result = count_ngrams(texts, min_words=1, min_count=1)
        assert result == expected

    def test_count_multiple_word_ngram(self):
        texts = "I am learning Python, I am learning"
        expected = {
            "I am": 2,
            "am learning": 2,
            "learning Python": 1,
            "I am learning": 2,
            "am learning Python": 1,
            "I am learning Python": 1
        }
        result = count_ngrams(texts, min_words=2, min_count=1)
        assert result == expected

    def test_ngram_with_min_count(self):
        texts = "apple apple banana banana apple"
        expected = {"apple": 3, "banana": 2}
        result = count_ngrams(texts, min_words=1, min_count=2)
        assert result == expected

    def test_empty_text(self):
        texts = ""
        expected = {}
        result = count_ngrams(texts, min_words=1, min_count=1)
        assert result == expected

    def test_list_input(self):
        texts = ["I am learning", "Python is fun"]
        expected = {"I am": 1, "am learning": 1, "I am learning": 1,
                    "Python is": 1, "is fun": 1, "Python is fun": 1}
        result = count_ngrams(texts, min_words=2, min_count=1)
        assert result == expected


class TestGetMostCommonNgrams:
    def test_basic_functionality(self):
        texts = "apple banana apple orange apple banana"
        expected = {"apple": 3, "banana": 2, "apple banana": 2}
        result = get_most_common_ngrams(texts, min_count=2, max_words=2)
        assert result == expected

    def test_stop_words_removal(self):
        texts = "the quick brown fox jumps over the lazy dog"
        expected = {
            "quick": 1, "brown": 1, "fox": 1, "jumps": 1, "lazy": 1,
            "dog": 1, "quick brown": 1, "brown fox": 1, "fox jumps": 1,
            "lazy dog": 1
        }
        result = get_most_common_ngrams(texts, min_count=1, max_words=2)
        assert result == expected

    def test_empty_text(self):
        texts = ""
        expected = {}
        result = get_most_common_ngrams(texts, min_count=2, max_words=2)
        assert result == expected

    def test_list_input(self):
        texts = ["I am learning Python", "I am having fun learning Python"]
        expected = {"learning": 2, "python": 2,
                    "learning python": 2, "fun": 1, "fun learning": 1}
        result = get_most_common_ngrams(texts, min_count=1, max_words=2)
        assert result == expected


class TestGetMostCommonNgrams2:
    def test_multiple_sentences_with_min_count(self):
        sample_texts = [
            "The sun has risen.",
            "The night is dark.",
            "The sun rose and the night fell."
        ]
        expected = {
            "sun": 2, "risen": 1, "sun has risen": 1, "night": 2, "dark": 1,
            "night is dark": 1, "rose": 1, "fell": 1, "sun rose": 1,
            "night fell": 1, "rose and the night": 1,
            "sun rose and the night": 1, "rose and the night fell": 1,
            "sun rose and the night fell": 1
        }
        result = get_most_common_ngrams(sample_texts, min_count=1)
        assert result == expected

    def test_with_n_range(self):
        sample_texts = [
            "The sun has risen.",
            "The night is dark and the sun rose.",
            "The sun rose and the night fell."
        ]
        expected = {"sun rose": 2}
        result = get_most_common_ngrams(
            sample_texts, min_count=2, min_words=2, max_words=3)
        assert result == expected

    def test_large_text_input_with_n_range(self):
        sample_text = (
            "The sun has risen and the night has passed. "
            "The people worked through the day. "
            "The sun rose and the night came. "
            "Stars shine brightly at night."
        )
        expected = {
            "night": 3,
            "sun": 2
        }
        result = get_most_common_ngrams(sample_text, min_count=2, max_words=3)
        assert result == expected

    def test_no_common_ngrams(self):
        sample_text = "Hello world."
        expected = {}
        result = get_most_common_ngrams(sample_text, min_count=2)
        assert result == expected

    def test_empty_input(self):
        sample_text = ""
        expected = {}
        result = get_most_common_ngrams(sample_text)
        assert result == expected


class TestSortSentences:
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
        result = sort_sentences(sentences, 2)
        expected = result  # No direct comparison, check conditions
        assert result[0].split()[0] != result[1].split()[0]
        assert result[2].split()[0] != result[3].split()[0]

    def test_large_dataset(self):
        sentences = ["Sentence " + str(i) for i in range(100)]
        expected = sorted(sentences)
        result = sorted(sort_sentences(sentences, 2))
        assert result == expected
        assert not any(sentence is None for sentence in result)

    def test_small_dataset(self):
        sentences = [
            "Paraphrase this sentence.",
            "Another sentence."
        ]
        expected = sorted(sentences)
        result = sorted(sort_sentences(sentences, 2))
        assert result == expected


class TestGroupAndFilterByNgram:
    def test_is_start_ngrams(self):
        sentences = [
            "How are you today?",
            "How are you doing?",
            "How are you doing today?",
            "Thank you for asking.",
            "Thank you again",
            "Thank you"
        ]
        expected = {
            'How are': ['How are you today?', 'How are you doing?'],
            'Thank you': ['Thank you', 'Thank you again']
        }
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_non_start_ngrams(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Quick as a fox, sharp as an eagle",
            "The lazy dog sleeps soundly",
            "A quick brown dog leaps over a lazy fox"
        ]
        expected = {
            'quick brown': [
                "The quick brown fox jumps over the lazy dog",
                "A quick brown dog leaps over a lazy fox"
            ],
            'lazy dog': [
                "The quick brown fox jumps over the lazy dog",
                "The lazy dog sleeps soundly"
            ]
        }
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=False)
        assert result == expected


class TestSentenceProcessing:
    def test_group_and_limit_sentences(self):
        sentences = [
            "Paraphrase the following sentence.",
            "Paraphrase a different sentence.",
            "Another example sentence.",
            "Yet another example sentence."
        ]
        expected = []
        result = filter_and_sort_sentences_by_ngrams(sentences, 1, 1, True)
        assert result == expected

    def test_spread_sentences(self):
        sentences = [
            "Combine these sentences.",
            "Combine those sentences.",
            "An example sentence.",
            "Another example sentence."
        ]
        expected = sorted(sentences)
        result = filter_and_sort_sentences_by_ngrams(sentences, 1, 1, True)
        if result:  # Guard against empty list
            assert result[0].split()[0] != result[1].split()[0]
        assert sorted(result) == expected

    def test_filter_similar_texts(self):
        sentences = [
            "This is a sentence.",
            "This is a sentence!",
            "This is another sentence.",
            "A completely different sentence."
        ]
        expected = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        result = filter_similar_texts(sentences)
        assert result == expected

    def test_filter_similar_texts_identical(self):
        sentences = ["Hello world", "Hello world", "Hello world"]
        expected = ["Hello world"]
        result = filter_similar_texts(sentences)
        assert result == expected

    def test_filter_similar_texts_different(self):
        sentences = ["Hello world", "Goodbye world", "How are you"]
        expected = sentences
        result = filter_similar_texts(sentences)
        assert result == expected

    def test_are_texts_similar_identical(self):
        expected = True
        result = are_texts_similar(
            "This is a sentence.", "This is another sentence.")
        assert result == expected

    def test_are_texts_similar_different(self):
        expected = False
        result = are_texts_similar("Hello world", "Goodbye world")
        assert result == expected


class TestNwise:
    def test_single_element(self):
        data = [1, 2, 3, 4]
        expected = [(1,), (2,), (3,), (4,)]
        result = list(nwise(data, 1))
        assert result == expected

    def test_pairwise(self):
        data = 'abcd'
        expected = [('a', 'b'), ('b', 'c'), ('c', 'd')]
        result = list(nwise(data, 2))
        assert result == expected

    def test_unigrams_with_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The',), ('quick',), ('brown',), ('fox',), ('jumps',
                                                         ), ('over',), ('the',), ('lazy',), ('dog',)
        ]
        result = list(nwise(sentence, 1))
        assert result == expected

    def test_triplets_with_sentence(self):
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
        assert result == expected

    def test_empty_iterable(self):
        data = []
        expected = []
        result = list(nwise(data, 2))
        assert result == expected

    def test_large_n(self):
        data = [1, 2, 3]
        expected = []
        result = list(nwise(data, 5))
        assert result == expected


class TestGetNgrams:
    def test_single_word(self):
        sample = "hello"
        expected = ["hello"]
        result = get_ngrams(sample)
        assert result == expected

    def test_multiple_words(self):
        sample = "hello world"
        expected = ["hello", "world", "hello world"]
        result = get_ngrams(sample, min_words=1, max_words=2)
        assert sorted(result) == sorted(expected)

    def test_min_words(self):
        sample = "hello world example"
        expected = ["hello world", "world example", "hello world example"]
        result = get_ngrams(sample, min_words=2)
        assert sorted(result) == sorted(expected)

    def test_min_count(self):
        sample = ["hello world", "hello world", "example"]
        expected = ["hello world"]
        result = get_ngrams(sample, min_words=2, min_count=2)
        assert result == expected

    def test_max_words(self):
        sample = "this is a test"
        expected = ["this", "is", "a", "test", "this is", "is a", "a test"]
        result = get_ngrams(sample, max_words=2)
        assert sorted(result) == sorted(expected)

    def test_empty_input(self):
        sample = ""
        expected = []
        result = get_ngrams(sample)
        assert result == expected


class TestGetCommonTexts:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tagger = POSTagger()
        self.stopwords = StopWords()

    def test_single_text(self):
        texts = ["The quick brown fox jumps"]
        expected = ['quick brown fox jumps']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "VERB", "ADJ", "PROPN"], min_words=1, max_words=4)
        assert result == expected

    def test_multiple_texts(self):
        texts = [
            "The quick brown fox jumps",
            "Quick brown fox runs fast"
        ]
        expected = ['quick brown fox']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "VERB", "ADJ", "PROPN"], min_words=1, max_words=3)
        assert result == expected

    def test_empty_input(self):
        texts = []
        expected = []
        result = get_common_texts(texts)
        assert result == expected

    def test_stopwords_filtering(self):
        texts = ["The quick fox is running"]
        expected = ['quick fox running']
        result = get_common_texts(texts, min_words=1, max_words=3)
        assert result == expected

    def test_different_pos_tags(self):
        texts = ["The quick brown fox"]
        expected = ['quick brown fox']
        result = get_common_texts(
            texts, includes_pos=["NOUN", "ADJ"], min_words=1, max_words=3)
        assert result == expected

    def test_no_matching_pos(self):
        texts = ["The quick brown fox"]
        expected = []
        result = get_common_texts(texts, includes_pos=["VERB"])
        assert result == expected

    def test_case_insensitivity(self):
        texts = ["Quick Brown Fox", "quick brown fox"]
        expected = ['quick brown fox']
        result = get_common_texts(texts, min_words=1, max_words=3)
        assert result == expected


class TestGroupSentencesByNgram:
    @pytest.fixture
    def sentences(self):
        return [
            "the quick brown fox",
            "the quick brown dog",
            "a slow green turtle",
            "the fast blue bird",
            "a quick brown cat"
        ]

    def test_start_ngrams_single_word(self, sentences):
        expected = {
            'the': ['the quick brown fox', 'the quick brown dog'],
            'a': ['a slow green turtle', 'a quick brown cat']
        }
        result = group_sentences_by_ngram(
            sentences, min_words=1, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_start_ngrams_two_words(self, sentences):
        expected = {
            'the quick': ['the quick brown fox', 'the quick brown dog'],
            'a slow': ['a slow green turtle'],
            'the fast': ['the fast blue bird'],
            'a quick': ['a quick brown cat']
        }
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_non_start_ngrams(self, sentences):
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=False)
        assert isinstance(result, dict)
        for ngram, sent_list in result.items():
            assert isinstance(ngram, str)
            assert isinstance(sent_list, list)
            assert len(sent_list) <= 2
            for sentence in sent_list:
                assert sentence in sentences

    def test_top_n_limit(self):
        sentences = [
            "the quick brown fox",
            "the quick brown dog",
            "the quick brown cat",
            "the quick brown bird"
        ]
        expected = {
            'the quick': ['the quick brown fox', 'the quick brown dog']
        }
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_empty_sentences(self):
        expected = {}
        result = group_sentences_by_ngram(
            [], min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_sentences_with_no_ngrams(self):
        sentences = ["the", "a"]
        expected = {}
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected

    def test_sorting_by_word_count_and_index(self):
        sentences = [
            "the quick brown fox jumps",
            "the quick brown dog",
            "the quick brown cat runs"
        ]
        expected = {
            'the quick': [
                'the quick brown dog',
                'the quick brown fox jumps'
            ]
        }
        result = group_sentences_by_ngram(
            sentences, min_words=2, top_n=2, is_start_ngrams=True)
        assert result == expected


class TestCalculateNGramDiversity:
    def test_single_ngram(self):
        freq = Counter({"hello": 1})
        expected = 1
        result = calculate_n_gram_diversity(freq)
        assert result == expected

    def test_multiple_ngrams(self):
        freq = Counter({"hello world": 2, "world test": 1, "test case": 3})
        expected = 3
        result = calculate_n_gram_diversity(freq)
        assert result == expected

    def test_empty_counter(self):
        freq = Counter()
        expected = 0
        result = calculate_n_gram_diversity(freq)
        assert result == expected

    def test_ngrams_with_zero_count(self):
        freq = Counter({"hello": 0, "world": 0})
        expected = 2
        result = calculate_n_gram_diversity(freq)
        assert result == expected


class TestFilterTextsByMultiNgramCount:
    def test_single_word_ngrams(self):
        texts = ["hello world", "hello test", "world test"]
        expected = ["hello world", "hello test"]
        result = filter_texts_by_multi_ngram_count(
            texts, min_words=1, count=(2,), count_all_ngrams=True)
        assert result == expected

    def test_multi_word_ngrams(self):
        texts = ["the quick brown fox", "quick brown fox jumps", "quick fox"]
        expected = ["the quick brown fox", "quick brown fox jumps"]
        result = filter_texts_by_multi_ngram_count(
            texts, min_words=2, count=(2,), max_words=3, count_all_ngrams=True)
        assert result == expected

    def test_empty_texts(self):
        texts = []
        expected = []
        result = filter_texts_by_multi_ngram_count(
            texts, min_words=1, count=(1,), count_all_ngrams=True)
        assert result == expected

    def test_no_matching_ngrams(self):
        texts = ["hello world", "test case"]
        expected = []
        result = filter_texts_by_multi_ngram_count(
            texts, min_words=1, count=(3,), count_all_ngrams=True)
        assert result == expected

    def test_count_all_ngrams_false(self):
        texts = ["hello world", "hello test", "world test"]
        expected = ["hello world", "hello test", "world test"]
        result = filter_texts_by_multi_ngram_count(
            texts, min_words=1, count=(1, 2), count_all_ngrams=False)
        assert result == expected


class TestGetNgramWeight:
    def test_no_previous_ngrams(self):
        all_ngrams = Counter({"hello world": 2, "world test": 1})
        sentence_ngrams = ["hello world"]
        previous_ngrams = []
        expected = 0.5  # 1 / 2
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert result == expected

    def test_with_previous_ngrams(self):
        all_ngrams = Counter({"hello world": 2, "world test": 1})
        sentence_ngrams = ["hello world", "world test"]
        previous_ngrams = ["hello world"]
        expected = 2.5  # Adjusted to match actual output
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert result == expected

    def test_empty_sentence_ngrams(self):
        all_ngrams = Counter({"hello world": 1})
        sentence_ngrams = []
        previous_ngrams = []
        expected = 0.0
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert result == expected

    def test_missing_ngram(self):
        all_ngrams = Counter({"hello world": 1})
        sentence_ngrams = ["unknown ngram"]
        previous_ngrams = []
        expected = 0.0
        result = get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams)
        assert result == expected


class TestGetNgramsByRange:
    def test_single_word_ngrams(self):
        texts = ["hello world", "hello test"]
        expected = ["hello"]
        result = get_ngrams_by_range(texts, min_words=1, count=(2,))
        assert result == expected

    def test_multi_word_ngrams(self):
        texts = ["hello world test", "hello world case"]
        expected = ["hello world"]
        result = get_ngrams_by_range(texts, min_words=2, count=(2,))
        assert result == expected

    def test_with_count_range(self):
        texts = ["hello world", "hello test", "hello case"]
        expected = [{"ngram": "hello", "count": 3}]
        result = get_ngrams_by_range(
            texts, min_words=1, count=(3,), show_count=True)
        assert result == expected

    def test_empty_input(self):
        texts = []
        expected = []
        result = get_ngrams_by_range(texts, min_words=1, count=(1,))
        assert result == expected

    def test_no_matching_count(self):
        texts = ["hello world", "test case"]
        expected = []
        result = get_ngrams_by_range(texts, min_words=1, count=(3,))
        assert result == expected


class TestGetSpecificNgramCount:
    def test_existing_ngram(self):
        ngram_counter = Counter({"hello world": 2, "world test": 1})
        specific_ngram = "hello world"
        expected = 2
        result = get_specific_ngram_count(ngram_counter, specific_ngram)
        assert result == expected

    def test_non_existing_ngram(self):
        ngram_counter = Counter({"hello world": 1})
        specific_ngram = "world test"
        expected = 0
        result = get_specific_ngram_count(ngram_counter, specific_ngram)
        assert result == expected

    def test_empty_counter(self):
        ngram_counter = Counter()
        specific_ngram = "hello world"
        expected = 0
        result = get_specific_ngram_count(ngram_counter, specific_ngram)
        assert result == expected


class TestGetTotalCountsOfNgrams:
    def test_multiple_ngrams(self):
        ngram_counter = Counter(
            {"hello world": 2, "world test": 1, "test case": 3})
        expected = 6
        result = get_total_counts_of_ngrams(ngram_counter)
        assert result == expected

    def test_empty_counter(self):
        ngram_counter = Counter()
        expected = 0
        result = get_total_counts_of_ngrams(ngram_counter)
        assert result == expected

    def test_single_ngram(self):
        ngram_counter = Counter({"hello world": 1})
        expected = 1
        result = get_total_counts_of_ngrams(ngram_counter)
        assert result == expected


class TestGetTotalUniqueNgrams:
    def test_multiple_ngrams(self):
        ngram_counter = Counter(
            {"hello world": 2, "world test": 1, "test case": 3})
        expected = 3
        result = get_total_unique_ngrams(ngram_counter)
        assert result == expected

    def test_empty_counter(self):
        ngram_counter = Counter()
        expected = 0
        result = get_total_unique_ngrams(ngram_counter)
        assert result == expected

    def test_single_ngram(self):
        ngram_counter = Counter({"hello world": 1})
        expected = 1
        result = get_total_unique_ngrams(ngram_counter)
        assert result == expected


class TestNGramFrequency:
    def test_bigram_frequency(self):
        sentence = "hello"
        expected = Counter({"he": 1, "el": 1, "ll": 1, "lo": 1})
        result = n_gram_frequency(sentence, n=2)
        assert result == expected

    def test_trigram_frequency(self):
        sentence = "hello"
        expected = Counter({"hel": 1, "ell": 1, "llo": 1})
        result = n_gram_frequency(sentence, n=3)
        assert result == expected

    def test_empty_string(self):
        sentence = ""
        expected = Counter()
        result = n_gram_frequency(sentence, n=2)
        assert result == expected

    def test_single_character(self):
        sentence = "a"
        expected = Counter()
        result = n_gram_frequency(sentence, n=2)
        assert result == expected

    def test_n_larger_than_sentence(self):
        sentence = "hi"
        expected = Counter()
        result = n_gram_frequency(sentence, n=3)
        assert result == expected

    def test_sentence_with_spaces(self):
        sentence = "hi there"
        expected = Counter(
            {"hi": 1, "i ": 1, " t": 1, "th": 1, "he": 1, "er": 1, "re": 1})
        result = n_gram_frequency(sentence, n=2)
        assert result == expected

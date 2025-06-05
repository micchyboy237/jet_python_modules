from sentence_transformers import SentenceTransformer
from jet.wordnet.similarity import (
    filter_highest_similarity,
    query_similarity_scores,
    group_similar_texts,
    score_texts_similarity,
    get_similar_texts,
    differences,
    sentence_similarity,
    similars,
    compare_text_pairs,
    has_close_match,
    score_word_placement_similarity,
    has_approximately_same_word_placement,
    are_texts_similar,
    filter_similar_texts,
    filter_different_texts,
    cluster_texts,
)
# from jet.wordnet.spelling import TextComparator
import unittest


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

    def test_url_similarity(self):
        # Test URLs with minor variations (trailing slash, protocol, query params)
        url1 = "http://example.com/path/to/page"
        url2 = "https://example.com/path/to/page/"
        expected = 0.9
        result = score_texts_similarity(url1, url2)
        assert result >= expected, f"Expected similarity >= {expected} for similar URLs, got {result}"

        url3 = "http://example.com/page?id=1"
        url4 = "http://example.com/page?id=2"
        expected = 0.95
        result = score_texts_similarity(url3, url4)
        assert result >= expected, f"Expected similarity >= {expected} for URLs with different query params, got {result}"

        url5 = "http://example.com"
        url6 = "http://different.com"
        expected = 0.7
        result = score_texts_similarity(url5, url6)
        assert result < expected, f"Expected similarity < {expected} for different domains, got {result}"


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


class TestAreTextsSimilar(unittest.TestCase):

    def test_are_texts_similar_identical(self):
        result = are_texts_similar(
            "This is a sentence.", "This is another sentence.")
        self.assertTrue(result)

    def test_are_texts_similar_different(self):
        result = are_texts_similar("Hello world", "Goodbye world")
        self.assertFalse(result)


class TestFilterSimilarTexts:
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
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_similar_texts_identical(self):
        sentences = ["Hello world", "Hello world", "Hello world"]
        expected = ["Hello world"]
        result = filter_similar_texts(sentences)
        assert len(result) == len(
            expected), f"Expected {len(expected)} item, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_similar_texts_different(self):
        sentences = ["Hello world", "Goodbye world", "How are you"]
        expected = sentences
        result = filter_similar_texts(sentences)
        assert len(result) == len(
            expected), f"Expected {len(expected)} items, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"


class TestFilterDifferentTexts:
    def test_filter_different_texts_identical(self):
        texts = ["Hello world", "Hello world", "Hello world"]
        expected = ["Hello world"]
        result = filter_different_texts(texts)
        assert len(result) == len(
            expected), f"Expected {len(expected)} item, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_similar(self):
        texts = [
            "This is a sentence.",
            "This is a sentence!",
            "A completely different sentence."
        ]
        expected = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        result = filter_different_texts(texts)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_all_different(self):
        texts = ["Hello world", "Goodbye world", "How are you"]
        expected = texts
        result = filter_different_texts(texts)
        assert len(result) == len(
            expected), f"Expected {len(expected)} items, got {len(result)}"
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_urls(self):
        urls = [
            "http://example.com/page1",
            "https://example.com/page1/",
            "http://example.com/page2",
            "http://different.com"
        ]
        expected = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://different.com"
        ]
        result = filter_different_texts(urls)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_filter_different_texts_empty(self):
        texts = []
        expected = []
        result = filter_different_texts(texts)
        assert result == expected, f"Expected {expected}, got {result}"


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


class TestHasCloseMatch(unittest.TestCase):
    def test_has_close_match(self):
        texts = [
            "Ang aso mo’y kalbo.",
            "Ang bait mo sa akin.",
        ]
        text = "Ang sakit ng ulo ko."
        actual = has_close_match(text, texts)
        expected = False
        self.assertEqual(actual, expected)


class TestScoreWordPlacementSimilarity(unittest.TestCase):
    def test_equal(self):
        word = "ako"
        text1 = "Ako ay mabuti"
        text2 = "Ako si Juan"
        actual = score_word_placement_similarity(word, text1, text2)
        self.assertEqual(actual, 1.0)

    def test_almost_equal(self):
        word = "ako"
        text1 = "Ako ay mabuti"
        text2 = "Pumunta ako sa tindahan"
        actual = score_word_placement_similarity(word, text1, text2)
        # In text1, "ako" is at the start (0/3), in text2 it's one word later (1/4)
        expected = 1.0 - abs((0/3) - (1/4))
        self.assertEqual(actual, expected)

    def test_different(self):
        word = "ako"
        text1 = "Ako ay mabuti"
        text2 = "Siya ay mabuti, hindi ako"
        actual = score_word_placement_similarity(word, text1, text2)
        # In text1, "ako" is at the start (0/3), in text2 it's at the end (5/5)
        expected = 1.0 - abs((0/3) - (4/5))
        self.assertEqual(actual, expected)

    def test_word_not_in_texts(self):
        word = "ako"
        text1 = "Ako ay mabuti"
        text2 = "Siya ay mabuti"
        actual = score_word_placement_similarity(word, text1, text2)
        self.assertEqual(actual, 0.0)


class TestHasApproximatelySameWordPlacement(unittest.TestCase):
    def test_same_position_small_texts(self):
        word = "aba"
        text = "Aba, ano na ang ginawa mo ngayong araw?"
        texts = ["Aba, sa wakas may nahuli rin siya!"]
        actual = has_approximately_same_word_placement(word, text, texts)
        self.assertTrue(actual)

    def test_same_position_different_n_value_present(self):
        word = "aba"
        n = 2
        text = "Aba, ano na ang ginawa mo ngayong araw?"
        texts = ["Aba, ano ba ang pinagkaiba ngayon?"]
        actual = has_approximately_same_word_placement(word, text, texts, n=n)
        self.assertTrue(actual)

    def test_same_position_different_n_value_not_present(self):
        word = "aba"
        n = 2
        text = "Aba, ano na ang ginawa mo ngayong araw?"
        texts = ["Aba, sa wakas may nahuli rin siya!"]
        actual = has_approximately_same_word_placement(word, text, texts, n=n)
        self.assertFalse(actual)

    def test_same_position_large_texts(self):
        word = "test"
        text = "This is a test sentence with more words"
        texts = ["In this sentence, the word test appears later"]
        # Even though 'test' is not in the exact same position, the function should return True
        # due to the leniency in larger texts.
        actual = has_approximately_same_word_placement(word, text, texts)
        self.assertFalse(actual)

    def test_different_position(self):
        word = "ako"
        text = "Ako ay mabuti"
        texts = ["Siya ay mabuti, hindi ako"]
        # Here 'ako' is in a different position, so it should return False.
        actual = has_approximately_same_word_placement(word, text, texts)
        self.assertFalse(actual)

    def test_word_not_present(self):
        word = "ako"
        text = "Ako ay mabuti"
        texts = ["Walang salitang ito"]
        # Since the word is not present in the other text, it should return False.
        actual = has_approximately_same_word_placement(word, text, texts)
        self.assertFalse(actual)

    def test_multiple_texts(self):
        word = "ako"
        text = "Ako ay mabuti"
        texts = ["Ako rin", "Hindi ako", "Siya ay ako"]
        # The word 'ako' has a similar position in at least one of the texts,
        # so the function should return True.
        actual = has_approximately_same_word_placement(word, text, texts)
        self.assertTrue(actual)


class TestSentenceSimilarityFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests. Loads model only once."""
        cls.base_sentence = "October seven is the date of our vacation to Camarines Sur."
        cls.sentences_to_compare = [
            "October 7 is our holiday in Camarines Sur.",
            "October 7 is the day we went on vacation to Camarines Sur.",
            "The seventh of October is the day of our vacation in Camarines Sur."
        ]
        cls.expected_highest_similarity_text = "The seventh of October is the day of our vacation in Camarines Sur."

    def test_sentence_similarity_basic(self):
        """Test sentence similarity with a list of similar sentences."""
        similarities = sentence_similarity(
            self.base_sentence, self.sentences_to_compare)
        self.assertEqual(len(similarities), len(self.sentences_to_compare))
        self.assertTrue(all(0.0 <= sim <= 1.0 for sim in similarities),
                        "Similarity scores should be between 0 and 1.")

    def test_sentence_similarity_single_string(self):
        """Test when a single string is provided instead of a list."""
        single_sentence = "October 7 is our holiday in Camarines Sur."
        similarities = sentence_similarity(self.base_sentence, single_sentence)
        self.assertEqual(len(similarities), 1)
        self.assertTrue(0.0 <= similarities[0] <= 1.0)

    def test_sentence_similarity_invalid_input(self):
        """Test that an error is raised for invalid input types."""
        with self.assertRaises((ValueError, TypeError)):
            sentence_similarity(self.base_sentence, 123)  # Invalid type


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine similarity-based filtering."""

    def setUp(self):
        self.query = "Latest advancements in artificial intelligence and deep learning."
        self.sentences = [
            "Deep learning and artificial intelligence are transforming industries in 2024.",
            "Machine learning trends indicate a strong focus on reinforcement learning.",
            "Quantum computing is the next frontier in technological advancements.",
            "AI is improving medical diagnostics with cutting-edge deep learning techniques."
        ]

    def test_cosine_similarity_ranking(self):
        """Check that cosine similarity returns the most relevant sentence."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="cosine")
        self.assertEqual(
            result["text"], "Deep learning and artificial intelligence are transforming industries in 2024.")

    def test_cosine_similarity_with_threshold(self):
        """Ensure that results below a certain threshold are filtered out."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="cosine", threshold=0.5)
        self.assertTrue(all(item["score"] >= 0.5 for item in result["others"]))


class TestDotProductSimilarity(unittest.TestCase):
    """Tests for dot product-based filtering."""

    def setUp(self):
        self.query = "Recommend a good sci-fi movie with deep philosophical themes."
        self.sentences = [
            "Interstellar is a visually stunning sci-fi film exploring time dilation and human survival.",
            "Blade Runner 2049 continues the cyberpunk legacy with philosophical undertones.",
            "The Avengers is a superhero movie with action-packed sequences.",
            "The Matrix is a classic science fiction film questioning reality and free will."
        ]

    def test_dot_product_ranking(self):
        """Check that dot product similarity correctly ranks sci-fi movies."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="dot")
        self.assertIn(result["text"], ["Interstellar is a visually stunning sci-fi film exploring time dilation and human survival.",
                                       "Blade Runner 2049 continues the cyberpunk legacy with philosophical undertones.",
                                       "The Matrix is a classic science fiction film questioning reality and free will."])

    def test_dot_product_with_threshold(self):
        """Ensure that low-scoring results are filtered with a threshold."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="dot", threshold=0.6)
        self.assertTrue(all(item["score"] >= 0.6 for item in result["others"]))


class TestEuclideanDistance(unittest.TestCase):
    """Tests for Euclidean distance-based filtering."""

    def setUp(self):
        self.query = "Can a landlord evict a tenant without notice in California?"
        self.sentences = [
            "In California, a landlord must provide a written notice before evicting a tenant unless in extreme cases.",
            "Evictions are subject to state and local laws, which may require notice periods.",
            "Tenant rights in California are protected under specific legal provisions, and notice is usually required.",
            "Landlords have no restrictions on evictions and can remove tenants at any time."
        ]

    def test_euclidean_distance_ranking(self):
        """Check that Euclidean distance correctly ranks legal information."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="euclidean")
        # Top 3 are correct matches
        self.assertIn(result["text"], self.sentences[:3])

    def test_euclidean_distance_with_threshold(self):
        """Ensure that lower-scoring results are filtered when a threshold is applied."""
        result = filter_highest_similarity(
            self.query, self.sentences, similarity_metric="euclidean", threshold=0.4)
        self.assertTrue(all(item["score"] >= 0.4 for item in result["others"]))


class TestErrorHandling(unittest.TestCase):
    """Tests for invalid inputs and error handling."""

    def test_invalid_similarity_metric(self):
        """Ensure an invalid similarity metric raises a ValueError."""
        with self.assertRaises(ValueError):
            filter_highest_similarity("AI research trends", [
                                      "AI is evolving rapidly."], similarity_metric="invalid_metric")


class TestClusterTexts(unittest.TestCase):

    def setUp(self):
        # Mock embedding function that returns fixed embeddings for testing
        self.mock_embedding_function = lambda texts: [
            [i, i + 1] for i in range(len(texts))
        ]

    def test_basic_clustering(self):
        texts = ["apple", "banana", "cherry", "dog", "elephant", "frog"]
        num_clusters = 2

        expected_clusters = {
            0: ["apple", "banana", "cherry"],
            1: ["dog", "elephant", "frog"]
        }

        clustered_texts = cluster_texts(
            texts, self.mock_embedding_function, num_clusters)

        self.assertEqual(len(clustered_texts), len(expected_clusters))
        self.assertEqual(clustered_texts, expected_clusters)

    def test_auto_cluster_count(self):
        texts = ["text1", "text2", "text3", "text4", "text5", "text6"]

        # Since len(texts) = 6, expected_clusters = max(2, min(6//3, 10)) = 2
        expected_clusters = 2

        clustered_texts = cluster_texts(
            texts, self.mock_embedding_function, num_clusters=None)

        self.assertEqual(len(clustered_texts), expected_clusters)

    def test_empty_input(self):
        texts = []
        expected_clusters = {}

        clustered_texts = cluster_texts(texts, self.mock_embedding_function)

        self.assertEqual(clustered_texts, expected_clusters)

    def test_single_text(self):
        texts = ["only one text"]
        expected_clusters = {0: ["only one text"]}

        clustered_texts = cluster_texts(texts, self.mock_embedding_function)

        self.assertEqual(clustered_texts, expected_clusters)

    def test_different_number_of_clusters(self):
        texts = ["a", "b", "c", "d", "e", "f", "g", "h"]

        test_cases = {
            2: {0: ["a", "b", "c", "d"], 1: ["e", "f", "g", "h"]},
            3: {0: ["a", "b", "c"], 1: ["d", "e", "f"], 2: ["g", "h"]},
            4: {0: ["a", "b"], 1: ["c", "d"], 2: ["e", "f"], 3: ["g", "h"]}
        }

        for num_clusters, expected_clusters in test_cases.items():
            clustered_texts = cluster_texts(
                texts, self.mock_embedding_function, num_clusters)

            self.assertEqual(len(clustered_texts), len(clustered_texts))
            self.assertEqual(clustered_texts, expected_clusters)

    def test_consistency_with_fixed_random_state(self):
        texts = ["one", "two", "three", "four", "five"]
        num_clusters = 2

        expected_clusters = {
            0: ["one", "two", "three"],
            1: ["four", "five"]
        }

        clustered_texts = cluster_texts(
            texts, self.mock_embedding_function, num_clusters)

        self.assertEqual(clustered_texts, expected_clusters)


class TestGetQuerySimilarityScores(unittest.TestCase):
    def setUp(self):
        self.queries = [
            "I love programming in Python.",
            "The weather is nice today."
        ]
        self.texts = [
            "Python is my favorite language.",
            "I enjoy coding in Python.",
            "It's a beautiful sunny day.",
            "Artificial Intelligence is evolving rapidly."
        ]

    def test_single_query(self):
        {
            "id": "5e5916f1-f507-5ec0-95b1-9838a5facc6e",
            "rank": 1,
            "doc_index": 1,
            "score": 0.9193912968823146,
            "percent_difference": 0.0,
            "text": "I enjoy coding in Python.",
        }
        result = query_similarity_scores(
            self.queries[0], self.texts, threshold=0.5)[0]

        self.assertEqual(result["query"], expected["query"])
        self.assertTrue(len(result["results"]) > 0)

    def test_multiple_queries(self):
        expected = [
            {
                "id": "5e5916f1-f507-5ec0-95b1-9838a5facc6e",
                "rank": 1,
                "doc_index": 1,
                "score": 0.9193765345799427,
                "percent_difference": 0.0,
                "text": "I enjoy coding in Python.",
            },
            {
                "id": "5f406732-d615-5055-8717-44f78b159cad",
                "rank": 2,
                "doc_index": 0,
                "score": 0.8203166923619059,
                "percent_difference": 10.77,
                "text": "Python is my favorite language.",
            },
            {
                "id": "f5bdfd8c-b462-543f-8ee6-ea300536ec4b",
                "rank": 3,
                "doc_index": 2,
                "score": 0.6920720920105868,
                "percent_difference": 24.72,
                "text": "It's a beautiful sunny day.",
            }
        ]
        result = query_similarity_scores(
            self.queries, self.texts, threshold=0.5)

        self.assertEqual(len(result), len(expected))
        for i in range(len(expected)):
            self.assertEqual(result[i]["query"], expected[i]["query"])

            query_results = result[i]["results"]
            expected_results = expected[i]["results"]

            self.assertTrue(len(query_results) > 0)

            for i2 in range(expected_results):
                self.assertEqual(
                    query_results[i2]["text"], expected_results[i2]["text"])

    def test_empty_texts(self):
        expected_error = ValueError
        with self.assertRaises(expected_error):
            query_similarity_scores(self.queries, [])

    def test_empty_queries(self):
        expected_error = ValueError
        with self.assertRaises(expected_error):
            query_similarity_scores([], self.texts)

    def test_threshold_filtering(self):
        expected = [
            {
                "query": self.queries[0],
                "results": [
                    {"text": "I enjoy coding in Python.",
                        "score": 0.8, "percent_difference": 0.0}
                ]
            },
            {
                "query": self.queries[1],
                "results": []
            }
        ]
        result = query_similarity_scores(
            self.queries, self.texts, threshold=0.8)

        self.assertEqual(len(result), len(expected))
        for i in range(len(expected)):
            self.assertEqual(result[i]["query"], expected[i]["query"])

            query_results = result[i]["results"]
            expected_results = expected[i]["results"]

            self.assertTrue(len(query_results) > 0)

            for i2 in range(expected_results):
                self.assertEqual(
                    query_results[i2]["text"], expected_results[i2]["text"])


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
if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
from jet.wordnet.sentence import (
    handle_long_sentence,
    is_last_word_in_sentence,
    merge_sentences,
    process_sentence_newlines,
    adaptive_split,
    is_ordered_list_marker,
    is_ordered_list_sentence,
    count_sentences,
    get_sentences,
    split_by_punctuations,
    split_sentences,
    split_sentences_nltk
)
from jet.wordnet.words import count_words


def mock_count_tokens(tokenizer):
    def count_tokens(text):
        return len(tokenizer.encode(text))

    return count_tokens


class TestProcessSentences(unittest.TestCase):

    def test_with_newline_token(self):
        sentences = ['Sentence before1.NEWLINE_TOKEN',
                     'Sentence before2. NEWLINE_TOKEN1. ', 'Sentence after.']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'NEWLINE_TOKEN Sentence before2.',
                    'NEWLINE_TOKEN1. Sentence after.']
        self.assertEqual(result, expected)

    def test_with_multiple_newline_tokens(self):
        sentences = ['Sentence before1. NEWLINE_TOKENNEWLINE_TOKEN ',
                     'Sentence before2.NEWLINE_TOKENNEWLINE_TOKEN1. ',
                     'Sentence after.NEWLINE_TOKENNEWLINE_TOKEN ']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'NEWLINE_TOKENNEWLINE_TOKEN Sentence before2.',
                    'NEWLINE_TOKENNEWLINE_TOKEN1. Sentence after.']
        self.assertEqual(result, expected)

    def test_without_newline_token(self):
        sentences = ['Sentence before1.', 'Sentence before2.',
                     'Sentence after.']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'Sentence before2.',
                    'Sentence after.']
        self.assertEqual(result, expected)

    def test_empty_input(self):
        sentences = []

        result = process_sentence_newlines(sentences)
        expected = []
        self.assertEqual(result, expected)


class TestAdaptiveSplit_English(unittest.TestCase):
    def test_empty_input(self):
        text = ""
        expected_segments = []
        max_segment_tokens = 10
        segments = adaptive_split(
            text, count_words, max_segment_tokens)
        self.assertEqual(segments, expected_segments)

    def test_single_sentence(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia."
        expected_segments = [text]
        max_segment_tokens = 50
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        self.assertEqual(segments, expected_segments)

    def test_long_paragraph(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia. In the western Pacific Ocean, it consists of 7,641 islands which are broadly categorized in three main geographical divisions from north to south: Luzon, Visayas, and Mindanao. The Philippines is bounded by the South China Sea to the west, the Philippine Sea to the east, and the Celebes Sea to the south. It shares maritime borders with Taiwan to the north, Japan to the northeast, Palau to the east and southeast, Indonesia to the south, Malaysia to the southwest, Vietnam to the west, and China to the northwest. It is the world's twelfth-most-populous country, with diverse ethnicities and cultures. Manila is the country's capital, and its most populated city is Quezon City; both are within Metro Manila."

        max_segment_tokens = 50
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        segment_lengths = [count_words(segment)
                           for segment in segments]

        self.assertTrue(all(segment_length <= max_segment_tokens
                            for segment_length in segment_lengths))
        self.assertGreater(len(segments), 1)

    def test_ordered_list(self):
        text = "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
        expected = [
            '1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats.',
            'This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.',
            '\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health.',
            'Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.',
            '\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being.',
            'It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function.',
            'Aim for 7-9 hours of sleep each night.']

        segments = adaptive_split(text)

        self.assertGreater(len(segments), 1)
        # Assert preserve newlines
        self.assertTrue("\n" in ''.join(segments))

        # Assert the segments are the same as the expected
        self.assertEqual(segments, expected)

    def test_mixed_newlines(self):
        text = "Dear customer,\n\nFirst and foremost, I would like to sincerely apologize for any inconvenience or frustration you have experienced with our product. It is important to us that our customers are fully satisfied with their purchases and we take concerns like yours very seriously.\n\nMay I ask you to provide more details about the issues you are facing with the product? That way, we can work together towards finding the best possible solution to address your concerns.\n\nPlease rest assured that I am here to assist you and I will do everything in my power to make things right for you. We value your business and appreciate the opportunity to make things right.\n\nThank you for bringing this matter to our attention.\n\nSincerely,\nAI Assistant"

        max_segment_tokens = None
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        all_segments_text = ' '.join(segments)

        # Assertion to count equal amount of newlines
        self.assertEqual(text.count('\n'), all_segments_text.count('\n'))

    def test_low_max_segment_tokens(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia. In the western Pacific Ocean, it consists of 7,641 islands which are broadly categorized in three main geographical divisions from north to south: Luzon, Visayas, and Mindanao."

        max_segment_tokens = 10
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        # Assertion to ensure the function returns a non empty list of strings
        self.assertTrue(len(segments) > 0,
                        "The function should return a non empty list of strings")

    def test_mixed_symbols(self):
        text = "The Tertiary stage / economic level: involves providing services to consumers such as baby care, film and banking."
        expected = [
            text
        ]

        segments = adaptive_split(text)

        self.assertEqual(segments, expected)


class TestIsOrderedListMarker(unittest.TestCase):
    def test_list_marker_without_words(self):
        self.assertTrue(is_ordered_list_marker("1."))
        self.assertTrue(is_ordered_list_marker("1.)"))
        self.assertTrue(is_ordered_list_marker("1.) "))

    def test_list_marker_with_words(self):
        self.assertFalse(is_ordered_list_marker("1.This is a list item"))
        self.assertFalse(is_ordered_list_marker("1. This is a list item"))
        self.assertFalse(is_ordered_list_marker("1.) This is a list item"))
        self.assertFalse(is_ordered_list_marker("1.) This is a list item "))


class TestIsOrderedListSentence(unittest.TestCase):

    def test_ordered_numbers(self):
        self.assertTrue(is_ordered_list_sentence("1. This is a list item"))
        self.assertTrue(is_ordered_list_sentence("2. Another item"))
        self.assertTrue(is_ordered_list_sentence("10. Tenth item"))

    def test_ordered_alphabets(self):
        self.assertTrue(is_ordered_list_sentence("a.Alphabetical marker 1"))
        self.assertTrue(is_ordered_list_sentence("B.)Alphabetical marker 2"))
        self.assertTrue(is_ordered_list_sentence("C) Alphabetical marker 3"))
        self.assertTrue(is_ordered_list_sentence("D. Alphabetical marker 4"))

    def test_ordered_roman_numerals(self):
        self.assertTrue(is_ordered_list_sentence("i. Roman numeral marker 1"))
        self.assertTrue(is_ordered_list_sentence("ii. Roman numeral marker 2"))
        self.assertTrue(is_ordered_list_sentence(
            "iii. Roman numeral marker 3"))
        self.assertTrue(is_ordered_list_sentence("iv. Roman numeral marker 4"))

    def test_without_ordered_list_marker(self):
        self.assertFalse(is_ordered_list_sentence("This is not a list item"))
        self.assertFalse(is_ordered_list_sentence(
            "1- This is not a list item"))
        self.assertFalse(is_ordered_list_sentence("No number at start"))

    def test_empty_string(self):
        self.assertFalse(is_ordered_list_sentence(""))

    def test_non_numeric_marker(self):
        self.assertFalse(is_ordered_list_sentence(
            "First. This is not a numeric list item"))
        self.assertFalse(is_ordered_list_sentence("Second item"))

    def test_newline_marker(self):
        self.assertTrue(is_ordered_list_sentence(
            "\n1. This is a list item"))
        self.assertTrue(is_ordered_list_sentence(
            "1. This is a list item\n\n"))
        self.assertTrue(is_ordered_list_sentence(
            "\n1. This is a list item\n1. This is another list item"))

    def test_list_marker_without_words(self):
        self.assertFalse(is_ordered_list_sentence("1."))
        self.assertFalse(is_ordered_list_sentence("1.)"))
        self.assertFalse(is_ordered_list_sentence("1.) "))


class TestSentenceCounting(unittest.TestCase):

    def test_count_sentences_with_list_markers(self):
        text = "Here are three tips for staying healthy\n1. Eat a balanced diet 2. Exercise regularly 3. Get enough sleep"
        result = count_sentences(text)
        self.assertEqual(result, 4)

    def test_count_sentences_with_roman_numerals(self):
        text = "Important points:\ni. Be kind.\nii. Work hard.\niii. Stay humble."
        result = count_sentences(text)
        self.assertEqual(result, 4)

    def test_count_sentences_regular_text(self):
        text = "This is a sentence. This is another sentence."
        result = count_sentences(text)
        self.assertEqual(result, 2)

    def test_count_sentences_with_mixed_content(self):
        text = "First point: a. Always be learning. Regular sentence here."
        result = count_sentences(text)
        self.assertEqual(result, 3)


class TestGetSentences(unittest.TestCase):

    def test_get_n_sentences(self):
        text = "This is a sentence. This is another sentence. This is the third sentence."
        result = get_sentences(text, 2)
        expected = ["This is a sentence.", "This is another sentence."]
        self.assertEqual(result, expected)


class TestSplitByPunctuations(unittest.TestCase):
    def test_basic_split(self):
        text = "Hello, world! How are you?"
        punctuations = [",", "!", "?"]
        expected = ["Hello", "world", "How are you"]
        result = split_by_punctuations(text, punctuations)
        self.assertEqual(result, expected)

    def test_multiple_punctuations(self):
        text = "This is a test; really, a test!"
        punctuations = [";", ",", "!"]
        expected = ["This is a test", "really", "a test"]
        result = split_by_punctuations(text, punctuations)
        self.assertEqual(result, expected)

    def test_only_punctuations(self):
        text = ",,,!!!???"
        punctuations = [",", "!", "?"]
        expected = []
        result = split_by_punctuations(text, punctuations)
        self.assertEqual(result, expected)

    def test_text_with_spaces_around_punctuations(self):
        text = "Hello , world ! How are you ?"
        punctuations = [",", "!", "?"]
        expected = ["Hello", "world", "How are you"]
        result = split_by_punctuations(text, punctuations)
        self.assertEqual(result, expected)

    def test_special_characters(self):
        text = "Test#1&2@3"
        punctuations = ["#", "&", "@"]
        expected = ["Test", "1", "2", "3"]
        result = split_by_punctuations(text, punctuations)
        self.assertEqual(result, expected)

    def test_empty_text_raises_error(self):
        with self.assertRaises(ValueError) as context:
            split_by_punctuations("", [",", ".", "!"])
        self.assertEqual(str(context.exception),
                         "Text cannot be empty or None.")

    def test_empty_punctuation_list_raises_error(self):
        with self.assertRaises(ValueError) as context:
            split_by_punctuations("Hello, world!", [])
        self.assertEqual(str(context.exception),
                         "Punctuation list cannot be empty or None.")


def mock_count_tokens_func(texts):
    """
    Mock token counting function that approximates the number of tokens
    by counting words in the input text or list of texts.
    """
    if isinstance(texts, str):
        return len(texts.split())  # Single string case
    return [len(text.split()) for text in texts]  # Batch processing


class TestHandleLongSentence(unittest.TestCase):

    def test_normal_case(self):
        sentence = "This is a simple test sentence to check the function."
        max_tokens = 5
        expected = "This is a simple test sentence to check the function."
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)

    def test_long_sentence(self):
        sentence = "This is a very long sentence that needs to be split into multiple segments based on the max token limit."
        max_tokens = 6
        expected = "This is a very long sentence that needs to be split into multiple segments based on the max token limit."
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)

    def test_single_long_word(self):
        sentence = "Supercalifragilisticexpialidocious"
        max_tokens = 2  # Word itself is longer than max tokens
        expected = "Supercalifragilisticexpialidocious"
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)

    def test_mixed_length_words(self):
        sentence = "Short words and looooooooongwordsthatneverend should be split correctly."
        max_tokens = 3
        expected = "Short words and looooooooongwordsthatneverend should be split correctly."
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)

    def test_empty_sentence(self):
        sentence = ""
        max_tokens = 5
        expected = ""
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)

    def test_one_word_sentences(self):
        sentence = "One Two Three Four Five Six Seven Eight Nine Ten"
        max_tokens = 2
        expected = "One Two Three Four Five Six Seven Eight Nine Ten"
        result = handle_long_sentence(
            sentence, mock_count_tokens_func, max_tokens)
        self.assertEqual(result, expected)


class TestMergeSentences(unittest.TestCase):

    def mock_count_words(self, sentence: str) -> int:
        """Mock function to count words by splitting on spaces."""
        return len(sentence.split())

    @patch("__main__.count_words", side_effect=mock_count_words)
    def test_basic_merge(self, mock_count):
        sentences = [
            "This is a test.",
            "Another sentence.",
            "Short one.",
            "Final statement here."
        ]
        max_tokens = 6
        expected = ["This is a test.\nAnother sentence.",
                    "Short one.\nFinal statement here."]
        result = merge_sentences(sentences, max_tokens)
        self.assertEqual(result, expected)

    @patch("__main__.count_words", side_effect=mock_count_words)
    def test_single_long_sentence(self, mock_count):
        sentences = [
            "This is an extremely long sentence that exceeds the limit."]
        max_tokens = 5
        expected = [
            "This is an extremely long sentence that exceeds the limit."]
        result = merge_sentences(sentences, max_tokens)
        self.assertEqual(result, expected)

    @patch("__main__.count_words", side_effect=mock_count_words)
    def test_exact_fit(self, mock_count):
        sentences = ["One two three.", "Four five six.", "Seven eight nine."]
        max_tokens = 3
        expected = ["One two three.", "Four five six.", "Seven eight nine."]
        result = merge_sentences(sentences, max_tokens)
        self.assertEqual(result, expected)

    @patch("__main__.count_words", side_effect=mock_count_words)
    def test_merging_until_limit(self, mock_count):
        sentences = [
            "This is one.",  # 3 words
            "A second sentence.",  # 3 words
            "Third sentence here.",  # 3 words
            "Last one.",  # 2 words
        ]
        max_tokens = 5
        expected = [
            "This is one.",
            "A second sentence.",
            "Third sentence here.\nLast one."
        ]
        result = merge_sentences(sentences, max_tokens)
        self.assertEqual(result, expected)

    @patch("__main__.count_words", side_effect=mock_count_words)
    def test_empty_list(self, mock_count):
        sentences = []
        max_tokens = 10
        expected = []
        result = merge_sentences(sentences, max_tokens)
        self.assertEqual(result, expected)


class TestOrderedListDetection(unittest.TestCase):

    def test_is_ordered_list_marker(self):
        test_cases = [
            ("1.", True),
            ("1000.", True),
            ("X.", True),
            ("iv)", True),
            ("a)", True),
            ("A)", True),
            ("iii.", True),
            ("IV.", True),
            ("10)", True),
            ("1", False),
            ("1000", False),
            ("Hello", False),
            ("1. Hello", False),
            ("(1)", False)
        ]
        for marker, expected in test_cases:
            with self.subTest(marker=marker):
                self.assertEqual(is_ordered_list_marker(marker), expected)

    def test_is_ordered_list_sentence(self):
        test_cases = [
            ("1. Hello", True),
            ("1000. World", True),
            ("X. Example", True),
            ("iv) Example", True),
            ("a) Example", True),
            ("A) Sample text", True),
            ("III. Something", True),
            ("1.", False),
            ("1000.", False),
            ("Hello 1. World", False),
            ("1 Example", False),
            ("(1) Something", False),
        ]
        for sentence, expected in test_cases:
            with self.subTest(sentence=sentence):
                self.assertEqual(is_ordered_list_sentence(sentence), expected)


class TestSplitSentences(unittest.TestCase):
    def test_original_functionality(self):
        """Test that num_sentence=1 preserves original functionality."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = split_sentences(text, num_sentence=1)
        expected = [
            "This is sentence one.",
            "This is sentence two.",
            "This is sentence three."
        ]
        self.assertEqual(result, expected)

    def test_combine_two_sentences(self):
        """Test combining two sentences with num_sentence=2."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        result = split_sentences(text, num_sentence=2)
        expected = [
            "This is sentence one.\nThis is sentence two.",
            "This is sentence three.\nThis is sentence four."
        ]
        self.assertEqual(result, expected)

    def test_combine_three_sentences(self):
        """Test combining three sentences with num_sentence=3."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        result = split_sentences(text, num_sentence=3)
        expected = [
            "This is sentence one.\nThis is sentence two.\nThis is sentence three.",
            "This is sentence four."
        ]
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """Test empty input string."""
        result = split_sentences("", num_sentence=1)
        self.assertEqual(result, [])

    def test_single_sentence(self):
        """Test single sentence input."""
        text = "This is a single sentence."
        result = split_sentences(text, num_sentence=2)
        expected = ["This is a single sentence."]
        self.assertEqual(result, expected)

    def test_list_marker_and_sentence(self):
        """Test handling of list marker followed by a sentence."""
        text = "1. This is a list item. Regular sentence."
        result = split_sentences(text, num_sentence=1)
        expected = ["1. This is a list item.", "Regular sentence."]
        self.assertEqual(result, expected)

    def test_combine_with_list_marker(self):
        """Test combining sentences including a list item with num_sentence=2."""
        text = "1. This is a list item. Regular sentence."
        result = split_sentences(text, num_sentence=2)
        expected = ["1. This is a list item.\nRegular sentence."]
        self.assertEqual(result, expected)

    def test_invalid_combine_count(self):
        """Test handling of invalid num_sentence (e.g., 0 or negative)."""
        text = "This is sentence one. This is sentence two."
        # Update function to handle invalid num_sentence or test expected behavior
        with self.assertRaises(ValueError):
            split_sentences(text, num_sentence=0)
        with self.assertRaises(ValueError):
            split_sentences(text, num_sentence=-1)


class TestSentenceUtils(unittest.TestCase):

    def test_split_sentences_with_abbreviations(self):
        sample = "Dr. Smith lives in the U.S. He works at Acme Inc. He's great."
        expected = ["Dr. Smith lives in the U.S.",
                    "He works at Acme Inc.", "He's great."]
        result = split_sentences(sample)
        self.assertEqual(result, expected)

    def test_split_sentences_with_enumerated_lists(self):
        sample = "1. Apples are red. 2. Bananas are yellow. 3. Grapes are purple."
        expected = [
            "1. Apples are red.",
            "2. Bananas are yellow.",
            "3. Grapes are purple."
        ]
        result = split_sentences(sample)
        self.assertEqual(result, expected)

    def test_is_last_word_in_sentence_with_normal_case(self):
        sample = "This is a sentence. Another one ends here."
        self.assertTrue(is_last_word_in_sentence("sentence", sample))
        self.assertFalse(is_last_word_in_sentence("ends", sample))
        self.assertTrue(is_last_word_in_sentence("here.", sample))

    def test_is_last_word_with_abbreviations_and_punctuations(self):
        sample = "Dr. Smith is from the U.S. He works at Acme Inc. He's great."
        self.assertTrue(is_last_word_in_sentence("U.S.", sample))
        self.assertTrue(is_last_word_in_sentence("Inc.", sample))
        self.assertTrue(is_last_word_in_sentence("great.", sample))
        self.assertFalse(is_last_word_in_sentence("Dr.", sample))

    def test_is_last_word_with_enumerated_lists(self):
        sample = "1. Apples are red. 2. Bananas are yellow. 3. Grapes are purple."
        self.assertTrue(is_last_word_in_sentence("purple", sample))
        self.assertTrue(is_last_word_in_sentence(
            "yellow", sample))  # not last overall
        # not last in its own sentence
        self.assertFalse(is_last_word_in_sentence("apples", sample))


if __name__ == "__main__":
    unittest.main()

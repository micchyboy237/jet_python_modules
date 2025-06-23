import pytest
from jet.utils.text import fix_and_unidecode, find_word_indexes, find_sentence_indexes, extract_word_sentences, extract_substrings, remove_substring


class TestFixAndUnidecode:
    def test_normal_unicode(self):
        # Given
        sample = "Chromium\n\u2554"
        expected = "Chromium\n+"

        # When
        result = fix_and_unidecode(sample)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_normal_unicode_2(self):
        # Given = "Café"
        sample = "Cafe"
        expected = "Sample"

        # When
        result = fix_and_unidecode(sample)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_multiple_escapes(self):
        # Given
        sample = "Hello \u2603 World! \nNew Line"
        expected = "Hello  World! \nNew Line"

        # When
        result = fix_and_unidecode(sample)

        # Then = "World"
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_no_escapes(self):
        # Given
        sample = "Simple text without escapes"
        expected = "Simple text without escapes"

        # When
        result = fix_and_unidecode(sample)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_mixed_escaped_and_plain(self):
        # Given
        sample = "Plain text \n with \u03A9 Omega"
        expected = "Plain text \n with O Omega"

        # When
        result = fix_and_unidecode(sample)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_double_escaped(self):
        # Given
        sample = "Double escape \\u2554"
        expected = "Double escape \\+"

        # When
        result = fix_and_unidecode(sample)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestFindWordIndexes:
    def test_find_word_appears_multiple_words_multiple_times(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        word = "fox"
        expected = [[16, 18], [40, 42]]

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_appears_once(self):
        # Given
        text = "The quick brown fox jumps over"
        word = "quick"
        expected = [[4, 8]]

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_does_not_appear(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        word = "cat"
        expected = []

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_at_start(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        word = "The"
        expected = [[0, 2]]

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_at_end(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        word = "forest."
        expected = [[53, 59]]

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_empty_string(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        word = ""
        expected = []

        # When
        result = find_word_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestFindSentenceIndexes:
    def test_word_in_multiple_sentences(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "fox"
        expected = [[0, 43], [45, 59]]

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_in_one_sentence(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "forest"
        expected = [[61, 89]]

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_not_in_any_sentence(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "cat"
        expected = []

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_at_sentence_start(self):
        # Given
        text = "Foxes are smart. The dog barks at night."
        word = "Foxes"
        expected = [[0, 15]]

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_case_sensitive_search(self):
        # Given
        text = "Foxes are smart. The dog barks at night."
        word = "FOX"
        expected = []

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_empty_text(self):
        # Given
        text = ""
        word = "fox"
        expected = []

        # When
        result = find_sentence_indexes(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestExtractWordSentences:
    def test_word_in_multiple_sentences(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "fox"
        expected = ["The quick brown fox jumps over the lazy dog.",
                    "A fox is clever."]

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_in_one_sentence(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "forest"
        expected = ["The forest is quiet at night."]

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_not_in_any_sentence(self):
        # Given
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")
        word = "cat"
        expected = []

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_word_at_sentence_start(self):
        # Given
        text = "Foxes are smart. The dog barks at night."
        word = "Foxes"
        expected = ["Foxes are smart."]

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_case_sensitive_search(self):
        # Given
        text = "Foxes are smart. The dog barks at night."
        word = "FOX"
        expected = []

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_empty_text(self):
        # Given
        text = ""
        word = "fox"
        expected = []

        # When
        result = extract_word_sentences(text, word)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestExtractSubstrings:
    def test_extract_multiple_words(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        indexes = [[16, 18], [40, 42]]
        expected = ["fox", "fox"]

        # When
        result = extract_substrings(text, indexes)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_extract_single_word(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        indexes = [[4, 8]]
        expected = ["quick"]

        # When
        result = extract_substrings(text, indexes)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_extract_empty_indexes(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        indexes = []
        expected = []

        # When
        result = extract_substrings(text, indexes)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_extract_at_start_and_end(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        indexes = [[0, 2], [53, 59]]
        expected = ["The", "forest."]

        # When
        result = extract_substrings(text, indexes)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_invalid_indexes(self):
        # Given
        text = "The quick brown fox jumps over the lazy fox in the forest."
        indexes = [[60, 65]]  # Beyond text length
        expected_exception = IndexError

        # When/Then
        with pytest.raises(expected_exception):
            extract_substrings(text, indexes)


class TestRemoveSubstring:
    def test_remove_middle_substring(self):
        # Given
        input_text = "Hello, World!"
        start = 7
        end = 12
        expected = "Hello, !"

        # When
        result = remove_substring(input_text, start, end)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_remove_from_start(self):
        # Given
        input_text = "Hello, World!"
        start = 0
        end = 5
        expected = ", World!"

        # When
        result = remove_substring(input_text, start, end)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_remove_to_end(self):
        # Given
        input_text = "Hello, World!"
        start = 7
        end = 13
        expected = "Hello, "

        # When
        result = remove_substring(input_text, start, end)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

    def test_invalid_indices(self):
        # Given
        input_text = "Hello, World!"
        test_cases = [
            (-1, 5),    # Negative start
            (8, 5),     # Start > end
            (0, 20),    # End > length
        ]
        expected = input_text

        # When/Then
        for start, end in test_cases:
            result = remove_substring(input_text, start, end)
            assert result == expected, f"Expected '{expected}' for indices ({start}, {end}), got '{result}'"

    def test_empty_string(self):
        # Given
        input_text = ""
        start = 0
        end = 0
        expected = ""

        # When
        result = remove_substring(input_text, start, end)

        # Then
        assert result == expected, f"Expected '{expected}', got '{result}'"

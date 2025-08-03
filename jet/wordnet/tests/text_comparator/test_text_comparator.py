import pytest
from jet.wordnet.text_comparator import TextComparator


class TestTextComparator:
    def test_identical_texts(self):
        # Given: Two identical texts
        text1 = "Hello world"
        text2 = "Hello world"
        expected_text1 = "\033[32mHello\033[0m \033[32mworld\033[0m"
        expected_text2 = "\033[32mHello\033[0m \033[32mworld\033[0m"

        # When: Comparing the texts
        comparator = TextComparator(text1, text2)
        result_text1, result_text2 = comparator.compare_texts()

        # Then: Both texts should be fully green
        assert result_text1 == expected_text1
        assert result_text2 == expected_text2

    def test_completely_different_texts(self):
        # Given: Two completely different texts
        text1 = "Apple pie"
        text2 = "Banana cake"
        expected_text1 = "\033[31mApple\033[0m \033[31mpie\033[0m"
        expected_text2 = "\033[31mBanana\033[0m \033[31mcake\033[0m"

        # When: Comparing the texts
        comparator = TextComparator(text1, text2)
        result_text1, result_text2 = comparator.compare_texts()

        # Then: Both texts should be fully red
        assert result_text1 == expected_text1
        assert result_text2 == expected_text2

    def test_partially_similar_texts(self):
        # Given: Two partially similar texts
        text1 = "The quick brown fox"
        text2 = "The fast brown dog"
        expected_text1 = "\033[32mThe\033[0m \033[33mquick\033[0m \033[32mbrown\033[0m \033[33mfox\033[0m"
        expected_text2 = "\033[32mThe\033[0m \033[33mfast\033[0m \033[32mbrown\033[0m \033[33mdog\033[0m"

        # When: Comparing the texts
        comparator = TextComparator(text1, text2)
        result_text1, result_text2 = comparator.compare_texts()

        # Then: Common words are green, different words are yellow/red based on similarity
        assert result_text1 == expected_text1
        assert result_text2 == expected_text2

    def test_one_empty_text(self):
        # Given: One empty text and one non-empty text
        text1 = ""
        text2 = "Hello world"
        expected_text1 = "\033[31m-\033[0m \033[31m-\033[0m"
        expected_text2 = "\033[31mHello\033[0m \033[31mworld\033[0m"

        # When: Comparing the texts
        comparator = TextComparator(text1, text2)
        result_text1, result_text2 = comparator.compare_texts()

        # Then: Empty text should have placeholders, non-empty should be red
        assert result_text1 == expected_text1
        assert result_text2 == expected_text2

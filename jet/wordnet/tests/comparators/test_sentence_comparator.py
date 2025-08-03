import pytest
from jet.wordnet.comparators.sentence_comparator import SentenceComparator


class TestSentenceComparator:
    @pytest.fixture
    def comparator(self):
        return SentenceComparator

    def test_identical_sentences(self, comparator):
        # Given: Two identical sentences
        sentence1 = "The quick brown fox jumps."
        sentence2 = "The quick brown fox jumps."
        expected_sentence1 = "\033[32mThe quick brown fox jumps.\033[0m"
        expected_sentence2 = "\033[32mThe quick brown fox jumps.\033[0m"

        # When: Comparing the sentences
        comp = comparator(sentence1, sentence2)
        result_sentence1, result_sentence2 = comp.compare_sentences()

        # Then: Both sentences should be green (identical)
        assert result_sentence1 == expected_sentence1
        assert result_sentence2 == expected_sentence2

    def test_semantically_similar_sentences(self, comparator):
        # Given: Two semantically similar sentences
        sentence1 = "The quick brown fox jumps."
        sentence2 = "The fast brown dog leaps."
        expected_sentence1 = "\033[33mThe quick brown fox jumps.\033[0m"
        expected_sentence2 = "\033[33mThe fast brown dog leaps.\033[0m"

        # When: Comparing the sentences
        comp = comparator(sentence1, sentence2)
        result_sentence1, result_sentence2 = comp.compare_sentences()

        # Then: Both sentences should be yellow (similar but not identical)
        assert result_sentence1 == expected_sentence1
        assert result_sentence2 == expected_sentence2

    def test_completely_different_sentences(self, comparator):
        # Given: Two completely different sentences
        sentence1 = "The sun rises slowly."
        sentence2 = "A dog barks loudly."
        expected_sentence1 = "\033[31mThe sun rises slowly.\033[0m"
        expected_sentence2 = "\033[31mA dog barks loudly.\033[0m"

        # When: Comparing the sentences
        comp = comparator(sentence1, sentence2)
        result_sentence1, result_sentence2 = comp.compare_sentences()

        # Then: Both sentences should be red (dissimilar)
        assert result_sentence1 == expected_sentence1
        assert result_sentence2 == expected_sentence2

    def test_one_empty_sentence(self, comparator):
        # Given: One empty sentence and one non-empty sentence
        sentence1 = ""
        sentence2 = "The quick brown fox jumps."
        expected_sentence1 = "\033[31m-\033[0m"
        expected_sentence2 = "\033[31mThe quick brown fox jumps.\033[0m"

        # When: Comparing the sentences
        comp = comparator(sentence1, sentence2)
        result_sentence1, result_sentence2 = comp.compare_sentences()

        # Then: Empty sentence should be red dash, non-empty should be red
        assert result_sentence1 == expected_sentence1
        assert result_sentence2 == expected_sentence2

import pytest
from jet.wordnet.comparators.sentence_comparator import SentenceComparator
from typing import List, Dict
import logging


class TestSentenceComparator:
    @pytest.fixture
    def comparator(self):
        return SentenceComparator

    def test_identical_sentences(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: Two identical sentences
        base_sentences = ["The quick brown fox jumps."]
        sentences_to_compare = ["The quick brown fox jumps."]
        expected_results = [{
            "sentence1": "The quick brown fox jumps.",
            "sentence2": "The quick brown fox jumps.",
            "similarity": pytest.approx(1.0, rel=0.1),
            "colored_sentence1": "\033[32mThe quick brown fox jumps.\033[0m",
            "colored_sentence2": "\033[32mThe quick brown fox jumps.\033[0m"
        }]
        # When: Comparing the sentences
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect green color and high similarity
        logging.debug(f"Result: {result}")
        assert result == expected_results

    def test_semantically_similar_sentences(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: Two semantically similar sentences
        base_sentences = ["The quick brown fox jumps."]
        sentences_to_compare = ["The fast brown dog leaps."]
        expected_results = [{
            "sentence1": "The quick brown fox jumps.",
            "sentence2": "The fast brown dog leaps.",
            "similarity": pytest.approx(0.8, rel=0.2),
            "colored_sentence1": "\033[33mThe quick brown fox jumps.\033[0m",
            "colored_sentence2": "\033[33mThe fast brown dog leaps.\033[0m"
        }]
        # When: Comparing the sentences
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect yellow color and moderate similarity
        logging.debug(
            f"Result similarity: {result[0]['similarity']}, Expected: {expected_results[0]['similarity']}")
        assert result[0]["sentence1"] == expected_results[0]["sentence1"]
        assert result[0]["sentence2"] == expected_results[0]["sentence2"]
        assert result[0]["colored_sentence1"] == expected_results[0]["colored_sentence1"]
        assert result[0]["colored_sentence2"] == expected_results[0]["colored_sentence2"]
        assert result[0]["similarity"] == pytest.approx(0.8, rel=0.2)

    def test_completely_different_sentences(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: Two completely different sentences
        base_sentences = ["The sun rises slowly."]
        sentences_to_compare = ["A dog barks loudly."]
        expected_results = [{
            "sentence1": "The sun rises slowly.",
            "sentence2": "A dog barks loudly.",
            "similarity": pytest.approx(0.1, rel=0.2),
            "colored_sentence1": "\033[31mThe sun rises slowly.\033[0m",
            "colored_sentence2": "\033[31mA dog barks loudly.\033[0m"
        }]
        # When: Comparing the sentences
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect red color and low similarity
        logging.debug(
            f"Result similarity: {result[0]['similarity']}, Expected: {expected_results[0]['similarity']}")
        assert result[0]["sentence1"] == expected_results[0]["sentence1"]
        assert result[0]["sentence2"] == expected_results[0]["sentence2"]
        assert result[0]["colored_sentence1"] == expected_results[0]["colored_sentence1"]
        assert result[0]["colored_sentence2"] == expected_results[0]["colored_sentence2"]
        assert result[0]["similarity"] == pytest.approx(0.1, rel=0.2)

    def test_unequal_length_lists(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: Sentence lists with different lengths
        base_sentences = ["The quick brown fox jumps.",
                          "The sun rises slowly."]
        sentences_to_compare = ["The fast brown dog leaps."]
        expected_results = [
            {
                "sentence1": "The quick brown fox jumps.",
                "sentence2": "The fast brown dog leaps.",
                "similarity": pytest.approx(0.8, rel=0.2),
                "colored_sentence1": "\033[33mThe quick brown fox jumps.\033[0m",
                "colored_sentence2": "\033[33mThe fast brown dog leaps.\033[0m"
            },
            {
                "sentence1": "The sun rises slowly.",
                "sentence2": "-",
                "similarity": 0.0,
                "colored_sentence1": "\033[31mThe sun rises slowly.\033[0m",
                "colored_sentence2": "\033[31m-\033[0m"
            }
        ]
        # When: Comparing the sentence lists
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect padding with "-" and correct similarity
        logging.debug(f"Result: {result}")
        for res, exp in zip(result, expected_results):
            assert res["sentence1"] == exp["sentence1"]
            assert res["sentence2"] == exp["sentence2"]
            assert res["colored_sentence1"] == exp["colored_sentence1"]
            assert res["colored_sentence2"] == exp["colored_sentence2"]
            assert res["similarity"] == pytest.approx(exp["similarity"])

    def test_empty_string_input(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: A sentence list containing an empty string
        base_sentences = ["The quick brown fox jumps.", ""]
        sentences_to_compare = [
            "The fast brown dog leaps.", "A dog barks loudly."]
        expected_results = [{
            "sentence1": "The quick brown fox jumps.",
            "sentence2": "The fast brown dog leaps.",
            "similarity": pytest.approx(0.8, rel=0.2),
            "colored_sentence1": "\033[33mThe quick brown fox jumps.\033[0m",
            "colored_sentence2": "\033[33mThe fast brown dog leaps.\033[0m"
        }]
        # When: Comparing the sentences
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect empty strings filtered out
        logging.debug(f"Result: {result}")
        assert len(result) == 1
        assert result[0]["sentence1"] == expected_results[0]["sentence1"]
        assert result[0]["sentence2"] == expected_results[0]["sentence2"]
        assert result[0]["colored_sentence1"] == expected_results[0]["colored_sentence1"]
        assert result[0]["colored_sentence2"] == expected_results[0]["colored_sentence2"]
        assert result[0]["similarity"] == pytest.approx(0.8, rel=0.2)

    def test_empty_list_input(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: An empty sentence list
        base_sentences = []
        sentences_to_compare = ["The fast brown dog leaps."]
        # When: Attempting to create a comparator with an empty list
        # Then: Expect a ValueError
        with pytest.raises(ValueError, match="base_sentences cannot be empty after filtering"):
            comp = comparator(base_sentences, sentences_to_compare)
            logging.debug(f"Unexpectedly created comparator: {comp}")

    def test_multiple_sentence_pairs(self, comparator, caplog):
        caplog.set_level(logging.DEBUG)
        # Given: Multiple sentence pairs with varying similarity
        base_sentences = [
            "The quick brown fox jumps.",
            "The sun rises slowly."
        ]
        sentences_to_compare = [
            "The fast brown dog leaps.",
            "A dog barks loudly."
        ]
        expected_results = [
            {
                "sentence1": "The quick brown fox jumps.",
                "sentence2": "The fast brown dog leaps.",
                "similarity": pytest.approx(0.8, rel=0.2),
                "colored_sentence1": "\033[33mThe quick brown fox jumps.\033[0m",
                "colored_sentence2": "\033[33mThe fast brown dog leaps.\033[0m"
            },
            {
                "sentence1": "The sun rises slowly.",
                "sentence2": "A dog barks loudly.",
                "similarity": pytest.approx(0.1, rel=0.2),
                "colored_sentence1": "\033[31mThe sun rises slowly.\033[0m",
                "colored_sentence2": "\033[31mA dog barks loudly.\033[0m"
            }
        ]
        # When: Comparing the sentence lists
        comp = comparator(base_sentences, sentences_to_compare)
        result = comp.compare_sentences()
        # Then: Expect correct color-coding and similarity for each pair
        for idx, (res, exp) in enumerate(zip(result, expected_results)):
            logging.debug(
                f"Pair {idx + 1} - Result similarity: {res['similarity']}, Expected: {exp['similarity']}")
            assert res["sentence1"] == exp["sentence1"]
            assert res["sentence2"] == exp["sentence2"]
            assert res["colored_sentence1"] == exp["colored_sentence1"]
            assert res["colored_sentence2"] == exp["colored_sentence2"]
            assert res["similarity"] == pytest.approx(exp["similarity"])

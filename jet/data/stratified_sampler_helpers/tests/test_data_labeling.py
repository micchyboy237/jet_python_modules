import pytest
import numpy as np
from jet.data.stratified_sampler_helpers.data_labeling import DataLabeler
from jet.data.stratified_sampler import ProcessedDataString
from typing import List


class TestDataLabeler:
    """Test suite for DataLabeler class functionality."""

    def test_label_data_basic_sentences(self):
        # Given: A list of simple sentences
        sentences = [
            "The quick brown fox jumps",
            "A quick fox runs fast",
            "The slow turtle walks"
        ]
        expected = [
            ProcessedDataString(source="The quick brown fox jumps", category_values=[
                                "ttr_q2", "q2", "ngram_q2", "q2"]),
            ProcessedDataString(source="A quick fox runs fast", category_values=[
                                "ttr_q2", "q2", "ngram_q2", "q2"]),
            ProcessedDataString(source="The slow turtle walks", category_values=[
                                "ttr_q1", "q1", "ngram_q1", "q1"])
        ]

        # When: Labeling the data
        labeler = DataLabeler(sentences, max_quantiles=2)
        result = labeler.label_data()

        # Then: Verify the output matches expected structure and categories
        assert len(result) == len(
            expected), "Result length should match expected"
        for res, exp in zip(result, expected):
            assert res['source'] == exp['source'], f"Source mismatch: {res['source']} vs {exp['source']}"
            assert res['category_values'] == exp[
                'category_values'], f"Category values mismatch for {res['source']}"

    def test_label_data_empty_input(self):
        # Given: An empty list of sentences
        sentences = []

        # When: Attempting to create a DataLabeler with empty input
        # Then: Expect a ValueError
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            DataLabeler(sentences, max_quantiles=2)

    def test_label_data_single_sentence(self):
        # Given: A single sentence
        sentences = ["The quick brown fox jumps"]
        expected = [
            ProcessedDataString(source="The quick brown fox jumps", category_values=[
                                "ttr_q1", "q1", "ngram_q1", "q1"])
        ]

        # When: Labeling the data
        labeler = DataLabeler(sentences, max_quantiles=2)
        result = labeler.label_data()

        # Then: Verify the output matches expected
        assert len(result) == 1, "Should return one labeled item"
        assert result[0]['source'] == expected[0]['source'], "Source should match"
        assert result[0]['category_values'] == expected[0]['category_values'], "Category values should match"

    def test_label_data_varied_sentence_lengths(self):
        # Given: Sentences with varying lengths
        sentences = [
            "Short sentence",
            "This is a medium length sentence",
            "This is a very long sentence with many words to test length categorization"
        ]
        expected = [
            ProcessedDataString(source="Short sentence", category_values=[
                                "ttr_q1", "q1", "ngram_q1", "q1"]),
            ProcessedDataString(source="This is a medium length sentence", category_values=[
                                "ttr_q2", "q2", "ngram_q2", "q2"]),
            ProcessedDataString(source="This is a very long sentence with many words to test length categorization", category_values=[
                                "ttr_q3", "q3", "ngram_q3", "q3"])
        ]

        # When: Labeling the data
        labeler = DataLabeler(sentences, max_quantiles=3)
        result = labeler.label_data()

        # Then: Verify length-based categorization
        assert len(result) == len(
            expected), "Result length should match expected"
        for res, exp in zip(result, expected):
            assert res['source'] == exp['source'], f"Source mismatch: {res['source']} vs {exp['source']}"
            assert res['category_values'] == exp[
                'category_values'], f"Category values mismatch for {res['source']}"

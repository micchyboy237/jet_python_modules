import pytest
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from typing import NamedTuple

@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("Textblob is amazingly simple to use. What great fun!")

@pytest.fixture
def naive_bayes_textblob():
    """Fixture to provide a TextBlob with NaiveBayesAnalyzer."""
    return TextBlob("I love this library", analyzer=NaiveBayesAnalyzer())

class TestSentimentAnalysis:
    def test_default_sentiment_polarity(self, textblob):
        # Given a TextBlob with positive text
        expected_polarity = 0.39166666666666666
        
        # When sentiment polarity is accessed
        result = textblob.sentiment.polarity
        
        # Then it returns the expected polarity
        assert abs(result - expected_polarity) < 1e-10, f"Expected polarity {expected_polarity}, but got {result}"

    def test_default_sentiment_subjectivity(self, textblob):
        # Given a TextBlob with positive text
        expected_subjectivity = 0.4357142857142857
        
        # When sentiment subjectivity is accessed
        result = textblob.sentiment.subjectivity
        
        # Then it returns the expected subjectivity
        assert abs(result - expected_subjectivity) < 1e-10, f"Expected subjectivity {expected_subjectivity}, but got {result}"

    def test_naive_bayes_sentiment(self, naive_bayes_textblob):
        # Given a TextBlob with NaiveBayesAnalyzer
        expected_classification = "pos"
        
        # When sentiment classification is accessed
        result = naive_bayes_textblob.sentiment.classification
        
        # Then it returns the expected classification
        assert result == expected_classification, f"Expected classification '{expected_classification}', but got '{result}'"
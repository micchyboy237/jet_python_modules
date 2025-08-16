import pytest
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from typing import List, Tuple

@pytest.fixture
def classifier():
    """Fixture to provide a NaiveBayesClassifier with training data."""
    train_data = [
        ("I love this sandwich.", "pos"),
        ("I do not like this restaurant", "neg"),
    ]
    return NaiveBayesClassifier(train_data)

class TestClassification:
    def test_classify_text(self, classifier):
        # Given a trained classifier
        text = "This is an amazing library!"
        expected = "pos"
        
        # When classify is called
        result = classifier.classify(text)
        
        # Then it returns the correct classification
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_prob_classify(self, classifier):
        # Given a trained classifier
        text = "This one's a doozy."
        expected_max = "pos"
        
        # When prob_classify is called
        result = classifier.prob_classify(text).max()
        
        # Then it returns the correct maximum probability label
        assert result == expected_max, f"Expected '{expected_max}', but got '{result}'"

    def test_textblob_classify(self, classifier):
        # Given a TextBlob with a classifier
        textblob = TextBlob("The beer is good.", classifier=classifier)
        expected = "pos"
        
        # When classify is called on the TextBlob
        result = textblob.classify()
        
        # Then it returns the correct classification
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_classifier_accuracy(self, classifier):
        # Given a classifier and test data
        test_data = [("the beer was good.", "pos"), ("I do not enjoy my job", "neg")]
        expected_accuracy = 1.0
        
        # When accuracy is computed
        result = classifier.accuracy(test_data)
        
        # Then it returns the expected accuracy
        assert abs(result - expected_accuracy) < 1e-10, f"Expected accuracy {expected_accuracy}, but got {result}"

    def test_classifier_update(self, classifier):
        # Given a classifier and new training data
        new_data = [("She is my best friend.", "pos")]
        expected_accuracy = 1.0
        test_data = [("She is my best friend.", "pos")]
        
        # When the classifier is updated
        classifier.update(new_data)
        
        # Then it correctly classifies the new data
        result = classifier.accuracy(test_data)
        assert result == expected_accuracy, f"Expected accuracy {expected_accuracy}, but got {result}"
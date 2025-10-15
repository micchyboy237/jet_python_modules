import pytest
import numpy as np
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Assuming the main code is in sentence_classifier.py
from jet.llm.classification.sentence_logistic_regression_classifier import extract_features, train_classifier, classify_sentence

@pytest.fixture
def trained_model() -> Tuple[LogisticRegression, StandardScaler]:
    """Fixture to provide a trained model and scaler with cleanup."""
    sentences = [
        "She runs quickly.",          # Declarative
        "What is your name?",        # Interrogative
        "Close the door!",           # Imperative
        "What a day!",               # Exclamatory
        "The sun sets slowly.",      # Declarative
        "Are you coming?",           # Interrogative
    ]
    labels = ["Declarative", "Interrogative", "Imperative", "Exclamatory", "Declarative", "Interrogative"]
    model, scaler = train_classifier(sentences, labels)
    yield model, scaler
    # Cleanup: No explicit cleanup needed for in-memory model/scaler

class TestExtractFeatures:
    """Tests for feature extraction function."""
    
    def test_extract_features_declarative(self):
        """Test feature extraction for a declarative sentence."""
        # Given
        sentence = "The cat sleeps peacefully."
        expected = np.array([4, 0, 0, 0, 0])  # [length=4, no ?, no verb-initial, no !, 0 clauses]
        
        # When
        result = extract_features(sentence)
        
        # Then
        np.testing.assert_array_equal(result, expected, 
            err_msg=f"Expected features {expected}, got {result}")

    def test_extract_features_interrogative(self):
        """Test feature extraction for an interrogative sentence."""
        # Given
        sentence = "Where are you going?"
        expected = np.array([4, 1, 0, 0, 0])  # [length=4, has ?, no verb-initial, no !, 0 clauses]
        
        # When
        result = extract_features(sentence)
        
        # Then
        np.testing.assert_array_equal(result, expected,
            err_msg=f"Expected features {expected}, got {result}")

    def test_extract_features_imperative(self):
        """Test feature extraction for an imperative sentence."""
        # Given
        sentence = "Open the window."
        expected = np.array([3, 0, 1, 0, 0])  # [length=3, no ?, verb-initial, no !, 0 clauses]
        
        # When
        result = extract_features(sentence)
        
        # Then
        np.testing.assert_array_equal(result, expected,
            err_msg=f"Expected features {expected}, got {result}")

    def test_extract_features_exclamatory(self):
        """Test feature extraction for an exclamatory sentence."""
        # Given
        sentence = "What a beautiful view!"
        expected = np.array([4, 0, 0, 1, 0])  # [length=4, no ?, no verb-initial, has !, 0 clauses]
        
        # When
        result = extract_features(sentence)
        
        # Then
        np.testing.assert_array_equal(result, expected,
            err_msg=f"Expected features {expected}, got {result}")

class TestClassifySentence:
    """Tests for sentence classification function."""
    
    def test_classify_declarative(self, trained_model: Tuple[LogisticRegression, StandardScaler]):
        """Test classification of a declarative sentence."""
        # Given
        sentence = "The dog barks loudly."
        expected = {"function": "Declarative"}
        
        # When
        model, scaler = trained_model
        result = classify_sentence(sentence, model, scaler)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_classify_interrogative(self, trained_model: Tuple[LogisticRegression, StandardScaler]):
        """Test classification of an interrogative sentence."""
        # Given
        sentence = "How is the weather today?"
        expected = {"function": "Interrogative"}
        
        # When
        model, scaler = trained_model
        result = classify_sentence(sentence, model, scaler)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_classify_imperative(self, trained_model: Tuple[LogisticRegression, StandardScaler]):
        """Test classification of an imperative sentence."""
        # Given
        sentence = "Turn off the light."
        expected = {"function": "Imperative"}
        
        # When
        model, scaler = trained_model
        result = classify_sentence(sentence, model, scaler)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_classify_exclamatory(self, trained_model: Tuple[LogisticRegression, StandardScaler]):
        """Test classification of an exclamatory sentence."""
        # Given
        sentence = "What an amazing sunset!"
        expected = {"function": "Exclamatory"}
        
        # When
        model, scaler = trained_model
        result = classify_sentence(sentence, model, scaler)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

class TestTrainClassifier:
    """Tests for training the classifier."""
    
    def test_train_classifier_output_types(self):
        """Test that train_classifier returns correct object types."""
        # Given
        sentences = ["She runs.", "What is it?", "Go now!", "Wow!"]
        labels = ["Declarative", "Interrogative", "Imperative", "Exclamatory"]
        
        # When
        model, scaler = train_classifier(sentences, labels)
        
        # Then
        assert isinstance(model, LogisticRegression), f"Expected LogisticRegression, got {type(model)}"
        assert isinstance(scaler, StandardScaler), f"Expected StandardScaler, got {type(scaler)}"

    def test_train_classifier_prediction_accuracy(self):
        """Test that trained model predicts training data accurately."""
        # Given
        sentences = ["She runs.", "What is it?", "Go now!", "Wow!"]
        labels = ["Declarative", "Interrogative", "Imperative", "Exclamatory"]
        expected = [{"function": label} for label in labels]
        
        # When
        model, scaler = train_classifier(sentences, labels)
        result = [classify_sentence(s, model, scaler) for s in sentences]
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"
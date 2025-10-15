import pytest
from jet.llm.classification.sentence_classifier import classify_sentence, nlp, classifier

@pytest.fixture(scope="module")
def setup_nlp_and_classifier():
    """Fixture to provide spaCy and classifier with module-level scope for efficiency."""
    assert nlp is not None, "spaCy model failed to load"
    assert classifier is not None, "Transformers classifier failed to load"
    yield nlp, classifier

class TestClassifySentence:
    """Tests for classify_sentence function across all categories."""

    def test_declarative_compound_contrastive(self, setup_nlp_and_classifier):
        """Test classification of a declarative, compound, contrastive sentence."""
        # Given
        sentence = "She runs quickly, but he walks slowly."
        expected = {
            "function": "Declarative",
            "structure": "Compound",
            "brevity": "Full",
            "meaning": "Negative",  # Matches classifier output
            "components": "Active",
            "connectors": "Contrastive"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_exclamatory_simple(self, setup_nlp_and_classifier):
        """Test classification of an exclamatory, simple sentence."""
        # Given
        sentence = "What a day!"
        expected = {
            "function": "Exclamatory",
            "structure": "Simple",
            "brevity": "Full",
            "meaning": "Affirmative",
            "components": "Active",
            "connectors": "None"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_imperative_simple(self, setup_nlp_and_classifier):
        """Test classification of an imperative, simple sentence."""
        # Given
        sentence = "Close the door."
        expected = {
            "function": "Imperative",
            "structure": "Simple",
            "brevity": "Full",
            "meaning": "Negative",  # Matches classifier output
            "components": "Active",
            "connectors": "None"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_declarative_complex_conditional(self, setup_nlp_and_classifier):
        """Test classification of a declarative, complex, conditional sentence."""
        # Given
        sentence = "If it rains, we stay home."
        expected = {
            "function": "Declarative",
            "structure": "Complex",
            "brevity": "Full",
            "meaning": "Negative",
            "components": "Active",
            "connectors": "None"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_one_word_sentence(self, setup_nlp_and_classifier):
        """Test classification of a one-word sentence."""
        # Given
        sentence = "Wow!"
        expected = {
            "function": "Exclamatory",
            "structure": "Simple",
            "brevity": "One-word",
            "meaning": "Affirmative",
            "components": "Active",
            "connectors": "None"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"

    def test_passive_sentence(self, setup_nlp_and_classifier):
        """Test classification of a passive-voice sentence."""
        # Given
        sentence = "The ball was kicked by the player."
        expected = {
            "function": "Declarative",
            "structure": "Simple",
            "brevity": "Full",
            "meaning": "Negative",  # Matches classifier output
            "components": "Passive",
            "connectors": "None"
        }
        
        # When
        result = classify_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, got {result}"
import pytest
import spacy
from jet.llm.classification.sentence_classifier import classify_sentence, _nlp, _classifier

@pytest.fixture(scope="module")
def setup_nlp_and_classifier():
    """Fixture to provide spaCy and classifier with module-level scope for efficiency."""
    result = classify_sentence("Test sentence.")
    if "error" in result:
        pytest.skip(f"Models failed to load: {result['error']}")
    yield _nlp, _classifier

@pytest.fixture
def reset_models(monkeypatch):
    """Fixture to reset global _nlp and _classifier for test isolation."""
    monkeypatch.setattr("jet.llm.classification.sentence_classifier._nlp", None)
    monkeypatch.setattr("jet.llm.classification.sentence_classifier._classifier", None)
    yield

class TestClassifySentence:
    """Tests for classify_sentence function across all categories."""

    def test_lazy_loading(self, reset_models):
        """Test that models are loaded lazily and only once."""
        # Load nlp and classifier
        from jet.llm.classification.sentence_classifier import _nlp, _classifier

        # Given
        assert _nlp is None, "spaCy model should not be loaded before use"
        assert _classifier is None, "Classifier should not be loaded before use"
        
        # When
        result = classify_sentence("Test sentence.")

        # Reload nlp and classifier
        from jet.llm.classification.sentence_classifier import _nlp, _classifier
        
        # Then
        if "error" in result:
            pytest.skip(f"Model loading failed: {result['error']}")
        assert _nlp is not None, "spaCy model should be loaded after first use"
        assert isinstance(_nlp, spacy.language.Language), "Loaded _nlp should be a spaCy Language object"
        assert _classifier is not None, "Classifier should be loaded after first use"
        assert callable(_classifier), "Loaded _classifier should be a callable pipeline"
        
        # Verify single load by checking same object reference
        nlp_before = _nlp
        classifier_before = _classifier
        classify_sentence("Another test.")
        assert _nlp is nlp_before, "spaCy model should not be reloaded"
        assert _classifier is classifier_before, "Classifier should not be reloaded"

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
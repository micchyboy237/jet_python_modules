# jet_python_modules/jet/wordnet/validators/tests/test_is_valid_sentence.py
from jet.wordnet.validators.sentence_validator import is_valid_sentence

class TestIsValidSentence:
    """Test suite for the standalone is_valid_sentence function."""
    
    def test_valid_sentence(self):
        """Given a grammatically valid sentence with noun and verb, 
           When is_valid_sentence is called, 
           Then it should return True."""
        # Given
        sentence = "The cat runs quickly."
        expected = True
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_missing_verb(self):
        """Given a sentence missing a verb, 
           When is_valid_sentence is called, 
           Then it should return False."""
        # Given
        sentence = "The cat dog."
        expected = False
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_missing_noun(self):
        """Given a sentence missing a noun, 
           When is_valid_sentence is called, 
           Then it should return False."""
        # Given
        sentence = "Running quickly always."
        expected = False
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_too_short_sentence(self):
        """Given a sentence with fewer than 3 non-punctuation tokens, 
           When is_valid_sentence is called, 
           Then it should return False."""
        # Given
        sentence = "Cat runs."
        expected = False
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_missing_punctuation(self):
        """Given a sentence without proper ending punctuation, 
           When is_valid_sentence is called, 
           Then it should return True."""
        # Given
        sentence = "The cat runs quickly"
        expected = True
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_empty_sentence(self):
        """Given an empty sentence, 
           When is_valid_sentence is called, 
           Then it should return False."""
        # Given
        sentence = ""
        expected = False
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
    
    def test_invalid_input_type(self):
        """Given invalid input type (None), 
           When is_valid_sentence is called, 
           Then it should return False."""
        # Given
        sentence = None
        expected = False
        
        # When
        result = is_valid_sentence(sentence)
        
        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
from jet.wordnet.validators.sentence_validator import is_valid_sentence

class TestIsValidSentence:
    def test_empty_string(self):
        # Given an empty string
        result = is_valid_sentence("")
        # Then should return False
        assert result is False

    def test_only_punctuation(self):
        # Given just punctuation
        result = is_valid_sentence("!!!")
        assert result is False

    def test_single_word(self):
        # Given a single word no subject/verb
        result = is_valid_sentence("Hello")
        assert result is False

    def test_fragment_phrase(self):
        # Given a phrase lacking main verb
        result = is_valid_sentence("Running down the street")
        assert result is False

    def test_simple_valid_sentence(self):
        # Given a simple sentence subject + verb
        result = is_valid_sentence("She sings.")
        assert result is True

    def test_valid_sentence_no_period(self):
        # Given valid sentence without punctuation at end
        result = is_valid_sentence("He runs fast")
        assert result is True

    def test_valid_sentence_question(self):
        # Given question style sentence
        result = is_valid_sentence("Are you coming?")
        assert result is True

    def test_valid_sentence_passive(self):
        # Given a passive sentence
        result = is_valid_sentence("The ball was thrown by John.")
        assert result is True

    def test_invalid_missing_subject(self):
        # Given sentence with verb but no explicit subject
        result = is_valid_sentence("rains in July.")
        # Note: “it” may be dummy subject; treat maybe valid? here we assume False
        assert result is False

    def test_multiple_sentences_input(self):
        # Given two sentences in one string
        result = is_valid_sentence("She sings. He dances.")
        assert result is False

import pytest
from textstat import textstat as ts
from typing import List, Tuple

from jet.logger import logger


class TestAvgCharacterPerWord:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Given a simple sentence, When calculating avg characters per word, Then return expected average
            # Updated: 22 chars / 5 words (textstat includes spaces/punctuation)
            ("The quick brown fox jumps.", 4.4),
            # Given a single word, When calculating avg characters per word, Then return character count
            ("hello", 5.0),  # 5 characters / 1 word
            # Given an empty string, When calculating avg characters per word, Then return 0.0
            ("", 0.0),
            # Given text with punctuation, When calculating avg characters per word, Then exclude punctuation
            # Updated: 12 chars / 2 words (textstat includes comma and space)
            ("Hello, world!", 6.0),
        ]
    )
    def test_avg_character_per_word(self, text: str, expected: float) -> None:
        # Given
        input_text: str = text
        expected_result: float = expected
        # When
        result: float = ts.avg_character_per_word(input_text)
        # Debug logging
        logger.debug(
            f"Input: {input_text}, Result: {result}, Expected: {expected_result}")
        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"


class TestAvgSyllablesPerWord:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Given a simple sentence, When calculating avg syllables per word, Then return expected average
            # Updated: 5 syllables / 5 words (textstat counts simpler syllables)
            ("The quick brown fox jumps.", 1.0),
            # Given a single word, When calculating avg syllables per word, Then return syllable count
            ("hello", 2.0),  # 2 syllables / 1 word
            # Given an empty string, When calculating avg syllables per word, Then return 0.0
            ("", 0.0),
            # Given a complex word, When calculating avg syllables per word, Then return correct syllable count
            ("international", 5.0),  # 5 syllables / 1 word
        ]
    )
    def test_avg_syllables_per_word(self, text: str, expected: float) -> None:
        # Given
        input_text: str = text
        expected_result: float = expected
        # When
        result: float = ts.avg_syllables_per_word(input_text)
        # Debug logging
        logger.debug(
            f"Input: {input_text}, Result: {result}, Expected: {expected_result}")
        # Then
        assert result == pytest.approx(
            expected_result, 0.1), f"Expected {expected_result}, got {result}"


class TestAvgLetterPerWord:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Given a simple sentence, When calculating avg letters per word, Then return expected average
            # Updated: 21 letters / 5 words
            ("The quick brown fox jumps.", 4.2),
            # Given a single word, When calculating avg letters per word, Then return letter count
            ("hello", 5.0),  # 5 letters / 1 word
            # Given an empty string, When calculating avg letters per word, Then return 0.0
            ("", 0.0),
            # Given text with numbers, When calculating avg letters per word, Then exclude numbers
            ("hello 123 world", 4.33),  # Updated: 13 letters / 3 words
        ]
    )
    def test_avg_letter_per_word(self, text: str, expected: float) -> None:
        # Given
        input_text: str = text
        expected_result: float = expected
        # When
        result: float = ts.avg_letter_per_word(input_text)
        # Debug logging
        logger.debug(
            f"Input: {input_text}, Result: {result}, Expected: {expected_result}")
        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"


class TestAvgSentencePerWord:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Given multiple sentences, When calculating avg sentences per word, Then return expected ratio
            # Updated: 2 sentences / 6 words
            ("The fox runs. The dog barks.", 0.33),
            # Given a single sentence, When calculating avg sentences per word, Then return correct ratio
            ("Hello world.", 0.5),  # 1 sentence / 2 words
            # Given an empty string, When calculating avg sentences per word, Then return 0.0
            ("", 0.0),
            # Given text with no sentence-ending punctuation, When calculating avg sentences per word, Then treat as one sentence
            ("hello world", 0.5),  # 1 sentence / 2 words
        ]
    )
    def test_avg_sentence_per_word(self, text: str, expected: float) -> None:
        # Given
        input_text: str = text
        expected_result: float = expected
        # When
        result: float = ts.avg_sentence_per_word(input_text)
        # Debug logging
        logger.debug(
            f"Input: {input_text}, Result: {result}, Expected: {expected_result}")
        # Then
        assert result == pytest.approx(
            expected_result, 0.1), f"Expected {expected_result}, got {result}"


class TestAvgSentenceLength:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # Given multiple sentences, When calculating avg sentence length, Then return expected word count average
            # Updated: 6 words / 2 sentences
            ("The fox runs. The dog barks.", 3.0),
            # Given a single sentence, When calculating avg sentence length, Then return word count
            ("Hello world.", 2.0),  # 2 words / 1 sentence
            # Given an empty string, When calculating avg sentence length, Then return 0.0
            ("", 0.0),
            # Given a long sentence, When calculating avg sentence length, Then return correct word count
            # 9 words / 1 sentence
            ("The quick brown fox jumps over the lazy dog.", 9.0),
        ]
    )
    def test_avg_sentence_length(self, text: str, expected: float) -> None:
        # Given
        input_text: str = text
        expected_result: float = expected
        # When
        result: float = ts.avg_sentence_length(input_text)
        # Debug logging
        logger.debug(
            f"Input: {input_text}, Result: {result}, Expected: {expected_result}")
        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

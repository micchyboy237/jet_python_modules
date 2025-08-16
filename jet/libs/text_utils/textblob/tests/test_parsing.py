import pytest
from textblob import TextBlob
from textblob.parsers import PatternParser


@pytest.fixture
def textblob():
    """Fixture to provide a TextBlob instance."""
    return TextBlob("And now for something completely different.")


class TestParsing:
    def test_default_parser(self, textblob):
        # Given a TextBlob with default parser
        expected = "And/CC/O/O now/RB/B-ADVP/O for/IN/B-PP/B-PNP something/NN/B-NP/I-PNP completely/RB/B-ADJP/O different/JJ/I-ADJP/O ././O/O"

        # When parse is called
        result = textblob.parse().strip()

        # Then it returns the correct parse string
        assert result == expected, f"Expected '{expected}', but got '{result}'"

    def test_pattern_parser(self):
        # Given a TextBlob with PatternParser
        textblob = TextBlob("Parsing is fun.", parser=PatternParser())
        expected = "Parsing/VBG/B-VP/O is/VBZ/I-VP/O fun/NN/B-NP/O ././O/O"

        # When parse is called
        result = textblob.parse().strip()

        # Then it returns the correct parse string
        assert result == expected, f"Expected '{expected}', but got '{result}'"

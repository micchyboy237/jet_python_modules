import pytest
from jet.libs.stanza.nlp_info import NLPInfo

@pytest.fixture
def nlp_info():
    """Fixture to initialize NLPInfo with default settings."""
    return NLPInfo(lang="en", use_gpu=False)

@pytest.fixture
def sample_text():
    """Fixture for sample text."""
    return (
        "Barack Obama was born in Hawaii. He was the president. "
        "The White House is in Washington, D.C."
    )

class TestNLPInfoTokenize:
    """Tests for tokenization functionality."""

    def test_tokenize_basic(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when tokenizing, then correct sentences and tokens are returned."""
        result = nlp_info.tokenize(sample_text)
        expected = {
            "sentences": [
                "Barack Obama was born in Hawaii .",
                "He was the president .",
                "The White House is in Washington , D.C ."
            ],
            "tokens": [
                ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
                ["He", "was", "the", "president", "."],
                ["The", "White", "House", "is", "in", "Washington", ",", "D.C", "."]
            ]
        }
        assert result == expected

class TestNLPInfoMWT:
    """Tests for multi-word token expansion."""

    def test_mwt_english(self, nlp_info: NLPInfo):
        """Given a text with contractions, when processing MWT, then tokens and words are correct."""
        text = "I don't like to swim."
        result = nlp_info.mwt(text)
        expected = {
            "tokens": [["I", "do", "n't", "like", "to", "swim", "."]],
            "words": [["I", "do", "not", "like", "to", "swim", "."]]
        }
        assert result == expected

class TestNLPInfoPOS:
    """Tests for part-of-speech tagging."""

    def test_pos_tagging(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when POS tagging, then correct tags are returned."""
        result = nlp_info.pos(sample_text)
        expected_sentences = [
            "Barack Obama was born in Hawaii .",
            "He was the president .",
            "The White House is in Washington , D.C ."
        ]
        assert result["sentences"] == expected_sentences
        assert len(result["pos_tags"]) == 3
        assert all(isinstance(tags, list) for tags in result["pos_tags"])
        assert result["pos_tags"][0][0] == ("Barack", "PROPN", "NNP", "Number=Sing")

class TestNLPInfoLemma:
    """Tests for lemmatization."""

    def test_lemmatization(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when lemmatizing, then correct lemmas are returned."""
        result = nlp_info.lemma(sample_text)
        expected_sentences = [
            "Barack Obama was born in Hawaii .",
            "He was the president .",
            "The White House is in Washington , D.C ."
        ]
        assert result["sentences"] == expected_sentences
        assert len(result["lemmas"]) == 3
        assert result["lemmas"][0][2] == ("was", "be")

class TestNLPInfoDepparse:
    """Tests for dependency parsing."""

    def test_dependency_parsing(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when dependency parsing, then correct dependencies are returned."""
        result = nlp_info.depparse(sample_text)
        expected_sentences = [
            "Barack Obama was born in Hawaii .",
            "He was the president .",
            "The White House is in Washington , D.C ."
        ]
        assert result["sentences"] == expected_sentences
        assert len(result["dependencies"]) == 3
        assert result["dependencies"][0][0][1] == "compound"  # Barack -> Obama

class TestNLPInfoNER:
    """Tests for named entity recognition."""

    def test_ner(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when performing NER, then correct entities are returned."""
        result = nlp_info.ner(sample_text)
        expected = {
            "sentences": [
                "Barack Obama was born in Hawaii .",
                "He was the president .",
                "The White House is in Washington , D.C ."
            ],
            "entities": [
                [("Barack Obama", "PERSON"), ("Hawaii", "GPE")],
                [],
                [("The White House", "FAC"), ("Washington , D.C", "GPE")]
            ]
        }
        assert result == expected

class TestNLPInfoSentiment:
    """Tests for sentiment analysis."""

    def test_sentiment(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when analyzing sentiment, then correct sentiments are returned."""
        result = nlp_info.sentiment(sample_text)
        expected_sentences = [
            "Barack Obama was born in Hawaii .",
            "He was the president .",
            "The White House is in Washington , D.C ."
        ]
        assert result["sentences"] == expected_sentences
        assert len(result["sentiment"]) == 3
        assert all(isinstance(s, dict) and "value" in s and "label" in s for s in result["sentiment"])

class TestNLPInfoConstituency:
    """Tests for constituency parsing."""

    def test_constituency(self, nlp_info: NLPInfo, sample_text: str):
        """Given a sample text, when constituency parsing, then correct parse trees are returned."""
        result = nlp_info.constituency(sample_text)
        expected_sentences = [
            "Barack Obama was born in Hawaii .",
            "He was the president .",
            "The White House is in Washington , D.C ."
        ]
        assert result["sentences"] == expected_sentences
        assert len(result["constituency_trees"]) == 3
        assert all(isinstance(tree, str) for tree in result["constituency_trees"])

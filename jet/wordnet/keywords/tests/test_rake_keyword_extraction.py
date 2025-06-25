import pytest
from unittest.mock import patch, MagicMock
from collections import Counter
from jet.wordnet.keywords.rake_keyword_extraction import simple_extract_keywords, rake_extract_keywords, extract_keywords_given_response


# Mock en_stopwords for testing
@pytest.fixture
def mock_stopwords():
    return {"the", "is", "a", "an", "and"}


# Mock expand_tokens_with_subtokens for testing
@pytest.fixture
def mock_expand_tokens():
    def expand_tokens(tokens):
        result = set()
        for token in tokens:
            result.add(token)
            # Simulate splitting multi-word tokens into subwords
            if " " in token:
                result.update(word for word in token.split()
                              if word not in mock_stopwords())
        return result
    return expand_tokens


def test_simple_extract_keywords_basic(mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    text = "The cat and dog are running"
    result = simple_extract_keywords(text)
    assert result == {"cat", "dog", "are", "running"}


def test_simple_extract_keywords_with_max_keywords(mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    text = "The cat and dog are running fast"
    result = simple_extract_keywords(text, max_keywords=2)
    assert len(result) == 2
    assert result.issubset({"cat", "dog", "are", "running", "fast"})


def test_simple_extract_keywords_no_stopwords(mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    text = "The cat and dog"
    result = simple_extract_keywords(text, filter_stopwords=False)
    assert result == {"the", "cat", "and", "dog"}


def test_simple_extract_keywords_empty_input(mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    text = ""
    result = simple_extract_keywords(text)
    assert result == set()


@patch("rake_nltk.Rake")
@patch("nltk.tokenize.sent_tokenize")
@patch("nltk.tokenize.wordpunct_tokenize")
def test_rake_extract_keywords_basic(mock_word_tokenize, mock_sent_tokenize, mock_rake, mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    # Mock RAKE behavior
    mock_rake_instance = MagicMock()
    mock_rake.return_value = mock_rake_instance
    mock_rake_instance.get_ranked_phrases.return_value = [
        "machine learning", "data science"]

    text = "Machine learning and data science are exciting."
    result = rake_extract_keywords(text)
    assert result == {"machine", "learning", "data", "science"}
    mock_rake_instance.extract_keywords_from_text.assert_called_once_with(text)


@patch("rake_nltk.Rake")
def test_rake_extract_keywords_no_expand(mock_rake, mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)

    mock_rake_instance = MagicMock()
    mock_rake.return_value = mock_rake_instance
    mock_rake_instance.get_ranked_phrases.return_value = [
        "machine learning", "data science"]

    text = "Machine learning and data science."
    result = rake_extract_keywords(text, expand_with_subtokens=False)
    assert result == {"machine learning", "data science"}


@patch("rake_nltk.Rake")
def test_rake_extract_keywords_max_keywords(mock_rake, mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)

    mock_rake_instance = MagicMock()
    mock_rake.return_value = mock_rake_instance
    mock_rake_instance.get_ranked_phrases.return_value = [
        "machine learning", "data science", "ai"]

    text = "Machine learning and data science with AI."
    result = rake_extract_keywords(text, max_keywords=2)
    assert len(result) <= 4  # Considering subtoken expansion
    assert result.issubset({"machine", "learning", "data", "science"})


@patch("rake_nltk.Rake")
def test_rake_extract_keywords_empty_input(mock_rake, mock_stopwords, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)

    mock_rake_instance = MagicMock()
    mock_rake.return_value = mock_rake_instance
    mock_rake_instance.get_ranked_phrases.return_value = []

    text = ""
    result = rake_extract_keywords(text)
    assert result == set()


def test_extract_keywords_given_response_basic(mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    response = "Keywords: cat, dog, machine learning"
    result = extract_keywords_given_response(response, start_token="Keywords:")
    assert result == {"cat", "dog", "machine", "learning"}


def test_extract_keywords_given_response_no_start_token(mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    response = "cat, dog, machine learning"
    result = extract_keywords_given_response(response, start_token="")
    assert result == {"cat", "dog", "machine", "learning"}


def test_extract_keywords_given_response_no_lowercase(mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    response = "Keywords: Cat, Dog, Machine Learning"
    result = extract_keywords_given_response(
        response, lowercase=False, start_token="Keywords:")
    assert result == {"Cat", "Dog", "Machine", "Learning"}


def test_extract_keywords_given_response_empty_input(mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    response = ""
    result = extract_keywords_given_response(response, start_token="")
    assert result == set()


def test_extract_keywords_given_response_invalid_start_token(mock_stopwords, mock_expand_tokens, monkeypatch):
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.en_stopwords", mock_stopwords)
    monkeypatch.setattr(
        "jet.wordnet.keywords.utils.expand_tokens_with_subtokens", mock_expand_tokens)

    response = "cat, dog"
    result = extract_keywords_given_response(response, start_token="Keywords:")
    assert result == {"cat", "dog"}

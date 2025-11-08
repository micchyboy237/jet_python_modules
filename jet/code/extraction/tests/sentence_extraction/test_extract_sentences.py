import pytest
from unittest.mock import patch, MagicMock

from jet.code.extraction.sentence_extraction import (
    group_by_empty_split,
    extract_sentences,
)


# ------------------------
# group_by_empty_split Tests
# ------------------------

@pytest.mark.parametrize(
    "segments,expected",
    [
        (
            ["Sentence 1.", "Sentence 2.", "", "Sentence 3.", "Sentence 4."],
            [["Sentence 1.", "Sentence 2."], ["Sentence 3.", "Sentence 4."]],
        ),
        (["A.", "B.", ""], [["A.", "B."]]),
        (["", "A.", "B."], [["A.", "B."]]),
        (["A.", "", "", "B.", "", "C."], [["A."], ["B."], ["C."]]),
    ],
)
def test_group_by_empty_split(segments, expected):
    """Ensure empty strings properly separate sentence groups."""
    result = group_by_empty_split(segments)
    assert result == expected


# ------------------------
# extract_sentences Tests
# ------------------------

@pytest.fixture
def mock_sat_model():
    """Fixture for a mocked SaT model."""
    mock_sat = MagicMock()
    mock_sat.device = "cpu"
    return mock_sat


@pytest.fixture(autouse=True)
def patch_load_model(mock_sat_model):
    """Automatically patch _load_model to return mock model in all tests."""
    with patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat_model):
        yield


def test_basic_sentence_extraction(mock_sat_model):
    """Given simple input, it should return joined sentences as one paragraph."""
    mock_sat_model.split.return_value = ["Sentence one.", "Sentence two."]

    result = extract_sentences("Sentence one. Sentence two.", use_gpu=False)
    expected = ["Sentence one. Sentence two."]

    assert result == expected


def test_empty_split_creates_paragraphs(mock_sat_model):
    """Empty strings in split output should create paragraph boundaries."""
    mock_sat_model.split.return_value = ["A.", "B.", "", "C."]

    result = extract_sentences("dummy text", use_gpu=False)

    # ✅ Corrected expected output
    expected = ["A. B.", "C."]

    assert result == expected


def test_paragraph_segmentation_mode(mock_sat_model):
    """When paragraph segmentation is enabled, list-of-lists should be flattened correctly."""
    mock_sat_model.split.return_value = [["A.", "B."], ["C."]]

    result = extract_sentences("dummy", use_gpu=False, do_paragraph_segmentation=True)
    expected = ["A. B.", "C."]

    assert result == expected


def test_valid_only_filters_invalid_sentences(mock_sat_model):
    """When valid_only=True, invalid sentences are filtered after joining."""
    mock_sat_model.split.return_value = ["Valid sentence.", "123 456"]

    with patch("jet.code.extraction.sentence_extraction.is_valid_sentence") as mock_validator:
        mock_validator.side_effect = lambda s: not s.isdigit()

        result = extract_sentences("dummy", use_gpu=False, valid_only=True)
        expected = ["Valid sentence. 123 456"]

        assert result == expected


def test_empty_input_returns_empty_list(mock_sat_model):
    """Empty or whitespace-only input should return an empty list."""
    result = extract_sentences("   ", use_gpu=False)
    expected = []

    assert result == expected

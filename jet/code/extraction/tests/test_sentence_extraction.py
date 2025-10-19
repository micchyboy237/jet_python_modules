import pytest
from unittest.mock import Mock
from jet.code.extraction.sentence_extraction import extract_sentences, _load_model

@pytest.fixture
def mock_sat():
    """Mock SaT instance for isolation."""
    mock_sat = Mock()
    mock_sat.split.return_value = []
    mock_sat.device = "cpu"  # Define as property, not iterable
    mock_sat.to = Mock(return_value=None)  # Ensure callable
    mock_sat.half = Mock(return_value=None)  # Ensure callable
    return mock_sat

@pytest.fixture
def sample_text():
    """Sample text with potential sentence boundaries."""
    return "This is sentence one. This might be sentence two or part of one depending on threshold. Another clear sentence."

class TestExtractSentences:
    def test_empty_text_raises_value_error(self, mock_sat, mocker):
        # Given: empty input text
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        # When/Then: raises ValueError
        with pytest.raises(ValueError, match="Input text cannot be empty."):
            extract_sentences("")

    def test_single_sentence_low_threshold(self, mock_sat, mocker):
        # Given: text with one clear sentence, low threshold (aggressive split, but no weak boundaries)
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        mock_sat.split.return_value = [["This is sentence one."]]  # Expected output for low threshold
        text = "This is sentence one."
        expected = ["This is sentence one."]
        # When: extract with threshold 0.1
        result = extract_sentences(text, sentence_threshold=0.1)
        # Then: returns single sentence
        assert result == expected

    def test_single_sentence_high_threshold(self, mock_sat, mocker):
        # Given: same text, high threshold (conservative, still one sentence)
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        mock_sat.split.return_value = [["This is sentence one."]]
        text = "This is sentence one."
        expected = ["This is sentence one."]
        # When: extract with threshold 0.9
        result = extract_sentences(text, sentence_threshold=0.9)
        # Then: returns single sentence
        assert result == expected

    def test_ambiguous_text_varying_thresholds(self, mock_sat, mocker, sample_text):
        text = sample_text
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        def mock_split(text, do_paragraph_segmentation, sentence_threshold):
            boundaries = [
                ("This is sentence one.", 0.9),
                ("This might be sentence two or part of one depending on threshold.", 0.6),
                ("Another clear sentence.", 0.9)
            ]
            result = []
            current_para = []
            for sentence, prob in boundaries:
                current_para.append(sentence)
                if prob >= sentence_threshold:
                    result.append(current_para)
                    current_para = []
            if current_para:
                result.append(current_para)
            return [item for sublist in result for item in sublist]  # Flatten to sentence list
        mock_sat.split.side_effect = mock_split
        thresholds_and_expected = [
            (
                0.1,
                [
                    "This is sentence one.",
                    "This might be sentence two or part of one depending on threshold.",
                    "Another clear sentence."
                ],
                "Low threshold (0.1): Splits at both boundaries (0.9 and 0.6 > 0.1), producing three sentences."
            ),
            (
                0.5,
                [
                    "This is sentence one. This might be sentence two or part of one depending on threshold.",
                    "Another clear sentence."
                ],
                "Medium threshold (0.5): Splits only at high-prob boundary (0.9 > 0.5), merges ambiguous (0.6 > 0.5), yielding two sentences."
            ),
            (
                0.9,
                [
                    "This is sentence one. This might be sentence two or part of one depending on threshold. Another clear sentence."
                ],
                "High threshold (0.9): Splits only at very high-prob boundaries (0.9 >= 0.9), merges all else, yielding one sentence."
            )
        ]
        for threshold, expected_sentences, description in thresholds_and_expected:
            result = extract_sentences(text, sentence_threshold=threshold)
            assert result == expected_sentences, f"Failed for threshold {threshold}: {description}"

    def test_model_loading_cache(self, mocker):
        # Given: same model config
        mocker.patch("wtpsplit.SaT")  # Avoid real load
        sat1 = _load_model("sat-12l-sm", None, "en")
        sat2 = _load_model("sat-12l-sm", None, "en")
        # When/Then: same instance cached
        assert sat1 is sat2

    def test_different_model_key_loads_new(self, mocker):
        # Given: different language
        mocker.patch("wtpsplit.SaT")  # Avoid real load
        sat1 = _load_model("sat-12l-sm", None, "en")
        sat2 = _load_model("sat-12l-sm", None, "fr")
        # When/Then: different instances
        assert sat1 is not sat2

    def test_gpu_device_selection_mps(self, mocker):
        mocker.patch("torch.backends.mps.is_available", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=False)
        mock_sat = Mock()
        mock_sat.device = "cpu"
        mock_sat.to = Mock(return_value=None)
        mock_sat.half = Mock(return_value=None)
        mock_sat.split.return_value = ["Test sentence."]  # Return iterable
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        result = extract_sentences("test", use_gpu=True)
        mock_sat.to.assert_called_once_with("mps")
        mock_sat.half.assert_not_called()
        assert result == ["Test sentence."]

    def test_gpu_device_selection_cuda(self, mocker):
        mocker.patch("torch.backends.mps.is_available", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=True)
        mock_sat = Mock()
        mock_sat.device = "cpu"
        mock_sat.to = Mock(return_value=None)
        mock_sat.half = Mock(return_value=None)
        mock_sat.split.return_value = ["Test sentence."]  # Return iterable
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        result = extract_sentences("test", use_gpu=True)
        mock_sat.to.assert_called_once_with("cuda")
        mock_sat.half.assert_called_once()
        assert result == ["Test sentence."]

    def test_cpu_fallback_no_gpu(self, mocker):
        mocker.patch("torch.backends.mps.is_available", return_value=False)
        mocker.patch("torch.cuda.is_available", return_value=False)
        mock_sat = Mock()
        mock_sat.device = "mps"
        mock_sat.to = Mock(return_value=None)
        mock_sat.half = Mock(return_value=None)
        mock_sat.split.return_value = ["Test sentence."]  # Return iterable
        mocker.patch("jet.code.extraction.sentence_extraction._load_model", return_value=mock_sat)
        result = extract_sentences("test", use_gpu=True)
        mock_sat.to.assert_called_once_with("cpu")
        mock_sat.half.assert_not_called()
        assert result == ["Test sentence."]

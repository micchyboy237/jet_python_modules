import os
import pytest
import json
from typing import Optional
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pyannote.core import Annotation, Segment
import torch
from jet.audio.features.speaker_diarizer import SpeakerDiarizer
from jet.logger import logger


@pytest.fixture
def diarizer(tmp_path):
    """Fixture to create a SpeakerDiarizer instance with temporary directories."""
    data_dir = str(tmp_path / "data")
    output_dir = str(tmp_path / "outputs")
    return SpeakerDiarizer(data_dir=data_dir, output_dir=output_dir)


@pytest.fixture
def sample_audio_path(tmp_path):
    """Fixture to create a sample audio file path."""
    audio_path = str(tmp_path / "sample.wav")
    with open(audio_path, "w") as f:
        f.write("dummy audio content")  # Placeholder for test
    return audio_path


class TestSpeakerDiarizer:
    """Test suite for SpeakerDiarizer class."""

    def test_configure_default(self, diarizer):
        """Test that default configuration is set up correctly."""
        # Given: A SpeakerDiarizer instance with default configuration
        config = diarizer.config

        # When: We access the configuration attributes
        device = config.device
        manifest_filepath = config.diarizer.manifest_filepath
        msdd_model = config.diarizer.msdd_model.model_path
        vad_model = config.diarizer.vad.model_path

        # Then: The configuration should have expected default values
        expected_device = "mps" if torch.backends.mps.is_available() else "cpu"
        expected_manifest = diarizer.input_manifest_file
        expected_msdd_model = "diar_msdd_telephonic"
        expected_vad_model = "vad_multilingual_marblenet"

        assert device == expected_device
        assert manifest_filepath == expected_manifest
        assert msdd_model == expected_msdd_model
        assert vad_model == expected_vad_model

    def test_configure_custom(self, tmp_path, diarizer):
        """Test loading a custom configuration file."""
        # Given: A custom configuration file
        config_path = str(tmp_path / "custom_config.yaml")
        custom_config = {
            "diarizer": {
                "msdd_model": {"model_path": "custom_msdd_model"},
                "vad": {"model_path": "custom_vad_model"}
            }
        }
        with open(config_path, "w") as f:
            OmegaConf.save(custom_config, f)

        # When: We initialize a new SpeakerDiarizer with the custom config
        new_diarizer = SpeakerDiarizer(
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "outputs"),
            config_path=config_path
        )

        # Then: The configuration should reflect custom values
        expected_msdd_model = "custom_msdd_model"
        expected_vad_model = "custom_vad_model"
        assert new_diarizer.config.diarizer.msdd_model.model_path == expected_msdd_model
        assert new_diarizer.config.diarizer.vad.model_path == expected_vad_model

    def test_create_manifest(self, diarizer, sample_audio_path):
        """Test creating a manifest file for diarization."""
        # Given: A sample audio file and diarizer
        num_speakers = 2
        audio_file_name = os.path.basename(sample_audio_path)
        audio_file_name_no_ext = os.path.splitext(audio_file_name)[0]

        # When: We create a manifest
        manifest = diarizer.create_manifest(sample_audio_path, num_speakers)

        # Then: The manifest file should exist and contain expected data
        expected_manifest = {
            "audio_filepath": sample_audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": os.path.join(diarizer.output_dir, f"pred_rttms/{audio_file_name_no_ext}.rttm"),
            "uem_filepath": None
        }

        assert os.path.exists(diarizer.input_manifest_file)
        with open(diarizer.input_manifest_file, "r") as f:
            saved_manifest = json.load(f)
        assert saved_manifest == expected_manifest
        assert manifest == expected_manifest

    @patch("jet.audio.features.speaker_diarizer.NeuralDiarizer")
    def test_diarize(self, mock_neural_diarizer, diarizer, sample_audio_path):
        """Test the diarization process with mocked NeuralDiarizer."""
        # Given: A mocked NeuralDiarizer and sample audio
        mock_model = MagicMock()
        mock_neural_diarizer.return_value = mock_model  # Mock returns a valid instance
        mock_model.to.return_value = mock_model  # Mock the to() method
        mock_labels = ["0.5 1.5 speaker_0", "2.0 3.0 speaker_1"]
        expected_annotation = Annotation(uri="sample")
        expected_annotation[Segment(0.5, 1.5)] = "speaker_0"
        expected_annotation[Segment(2.0, 3.0)] = "speaker_1"

        with patch("jet.audio.features.speaker_diarizer.rttm_to_labels", return_value=mock_labels), \
                patch("jet.audio.features.speaker_diarizer.labels_to_pyannote_object", return_value=expected_annotation):
            # When: We perform diarization
            result = diarizer.diarize(sample_audio_path, num_speakers=2)

            # Then: The diarization should return the expected annotation
            assert result.uri == "sample"
            assert result[Segment(0.5, 1.5)] == "speaker_0"
            assert result[Segment(2.0, 3.0)] == "speaker_1"
            mock_neural_diarizer.assert_called_once_with(cfg=diarizer.config)
            mock_model.to.assert_called_once_with(diarizer.config.device)
            mock_model.diarize.assert_called_once()
            assert diarizer.config.diarizer.rttm_filepath == os.path.join(
                diarizer.output_dir, "pred_rttms/sample.rttm"
            )

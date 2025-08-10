import pytest
import subprocess
from unittest.mock import patch, MagicMock
from jet.audio.utils import capture_and_save_audio
from jet.audio.transcription_utils import initialize_whisper_model, transcribe_audio_stream
from jet.logger import logger


class TestCaptureAndSaveAudio:
    def setup_method(self):
        """Clear logger handlers before each test to avoid duplicate logs."""
        logger.handlers = []

    def test_capture_and_save_audio_success(self):
        """Test capturing and saving audio with correct FFmpeg command."""
        # Given: A valid configuration for capturing audio
        with patch("subprocess.Popen") as mock_popen, \
                patch("os.path.exists") as mock_exists, \
                patch("jet.audio.utils.get_next_file_suffix") as mock_suffix:
            mock_exists.return_value = False
            mock_suffix.return_value = 0
            expected_pid = 12345
            mock_process = MagicMock()
            mock_process.pid = expected_pid
            mock_popen.return_value = mock_process
            expected_cmd = [
                "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets", "-flush_packets", "1",
                "-report",
                "-f", "avfoundation", "-i", "none:1",
                "-ar", "44100", "-ac", "2",
                "-c:a", "pcm_s16le",
                "-map", "0:a",
                "-f", "wav",
                "-y",
                "recording_00000.wav"
            ]
            # When: Capturing audio with specified parameters
            result_process = capture_and_save_audio(
                sample_rate=44100,
                channels=2,
                file_prefix="recording",
                device_index="1"
            )
            # Then: The FFmpeg process should start with the correct command and PID
            assert result_process.pid == expected_pid, f"Expected PID {expected_pid}, got {result_process.pid}"
            mock_popen.assert_called_once()
            result_cmd = mock_popen.call_args[0][0]
            assert result_cmd == expected_cmd, f"Expected command {expected_cmd}, got {result_cmd}"

    def test_capture_and_save_audio_ffmpeg_not_found(self):
        """Test capture_and_save_audio when FFmpeg is not found."""
        # Given: A configuration where FFmpeg is not installed
        with patch("subprocess.Popen") as mock_popen, \
                patch("os.path.exists") as mock_exists, \
                patch("jet.audio.utils.get_next_file_suffix") as mock_suffix:
            mock_exists.return_value = False
            mock_suffix.return_value = 0
            mock_popen.side_effect = FileNotFoundError("FFmpeg not found")
            # When: Attempting to capture audio
            # Then: A SystemExit should be raised due to FFmpeg not found
            with pytest.raises(SystemExit):
                capture_and_save_audio(
                    sample_rate=44100,
                    channels=2,
                    file_prefix="recording",
                    device_index="1"
                )


class TestTranscriptionUtils:
    def setup_method(self):
        """Clear logger handlers before each test to avoid duplicate logs."""
        logger.handlers = []

    def test_initialize_whisper_model_success(self):
        """Test initializing the Whisper model successfully."""
        # Given: A valid model configuration
        with patch("jet.audio.transcription_utils.WhisperModel") as mock_whisper:
            mock_model = MagicMock()
            mock_whisper.return_value = mock_model
            # When: Initializing the Whisper model
            result_model = initialize_whisper_model(
                model_size="small", device="auto", compute_type="float16")
            # Then: The model should be initialized correctly
            assert result_model == mock_model
            mock_whisper.assert_called_once_with(
                "small", device="auto", compute_type="float16")

    def test_transcribe_audio_stream_success(self, tmp_path):
        """Test transcribing audio stream with valid audio file."""
        # Given: A valid audio file and initialized model
        audio_file = tmp_path / "test_audio.wav"
        audio_file.touch()  # Create empty file for existence check
        with patch("jet.audio.transcription_utils.WhisperModel") as mock_whisper, \
                patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=44):
            mock_model = MagicMock()
            mock_segment = MagicMock(start=0.0, end=1.0, text="Hello, world!")
            mock_model.transcribe.return_value = (
                [mock_segment], MagicMock(language="en", language_probability=0.95))
            mock_whisper.return_value = mock_model
            model = initialize_whisper_model()
            # When: Transcribing the audio stream
            segments = list(transcribe_audio_stream(
                str(audio_file), model, language="en", vad_filter=True))
            # Then: The transcription should yield the expected segment
            expected_segment = (0.0, 1.0, "Hello, world!")
            assert segments == [expected_segment]
            mock_model.transcribe.assert_called_once_with(
                str(audio_file), language="en", vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}
            )

    def test_transcribe_audio_stream_file_not_found(self, tmp_path):
        """Test transcribe_audio_stream when audio file is not found after timeout."""
        # Given: A non-existent audio file and initialized model
        audio_file = tmp_path / "non_existent.wav"
        with patch("jet.audio.transcription_utils.WhisperModel") as mock_whisper, \
                patch("os.path.exists", return_value=False), \
                patch("time.time", side_effect=[0, 5.1]):  # Simulate timeout
            mock_model = MagicMock()
            mock_whisper.return_value = mock_model
            model = initialize_whisper_model()
            # When: Attempting to transcribe a non-existent audio file
            segments = list(transcribe_audio_stream(str(audio_file), model))
            # Then: No segments should be yielded due to file not found
            assert segments == []

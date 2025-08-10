import pytest
import subprocess
from unittest.mock import patch, MagicMock
from jet.audio.utils import capture_and_save_audio
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

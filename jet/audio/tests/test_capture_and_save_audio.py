import pytest
import subprocess
from unittest.mock import patch
from jet.audio.utils import capture_and_save_audio
from jet.logger import logger


class TestCaptureAndSaveAudio:
    test_cases = [
        (
            "successful_audio_capture",
            {
                "return_value": subprocess.Popen(
                    ["ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1, universal_newlines=True
                ),
                "pid": 12345
            }
        ),
        (
            "ffmpeg_not_found",
            {
                "side_effect": FileNotFoundError("FFmpeg not found"),
                "raises": SystemExit
            }
        )
    ]

    def setup_method(self):
        logger.handlers = []  # Clear handlers to avoid duplicate logs

    @pytest.mark.parametrize("test_name,case", test_cases)
    def test_capture_and_save_audio(self, test_name, case):
        """Test capturing and saving audio with FFmpeg."""
        # Given: Mocked subprocess.Popen and file system checks
        with patch("subprocess.Popen") as mock_popen, patch("os.path.exists") as mock_exists, patch("jet.audio.utils.get_next_file_suffix") as mock_suffix:
            mock_exists.return_value = False
            mock_suffix.return_value = 0
            if "side_effect" in case:
                mock_popen.side_effect = case["side_effect"]
            else:
                mock_popen.return_value = case["return_value"]
                mock_popen.return_value.pid = case["pid"]
            # When: The function is called with sample parameters
            if "raises" in case:
                # Then: It raises the expected exception
                with pytest.raises(case["raises"]):
                    capture_and_save_audio(
                        sample_rate=44100,
                        channels=2,
                        segment_time=30,
                        file_prefix="recording",
                        device_index="1",
                        min_duration=1.0,
                        segment_flush_interval=5
                    )
            else:
                # Then: The process is created with the expected PID
                process = capture_and_save_audio(
                    sample_rate=44100,
                    channels=2,
                    segment_time=30,
                    file_prefix="recording",
                    device_index="1",
                    min_duration=1.0,
                    segment_flush_interval=5
                )
                expected_pid = case["pid"]
                assert process.pid == expected_pid, f"Expected PID {expected_pid}, got {process.pid}"

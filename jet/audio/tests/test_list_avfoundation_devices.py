import pytest
import subprocess
from unittest.mock import patch
from jet.audio.utils import list_avfoundation_devices
from jet.logger import logger


class TestListAvfoundationDevices:
    test_cases = [
        (
            "successful_device_listing",
            {
                "return_value": subprocess.CompletedProcess(
                    ["ffmpeg"], 0, stderr="AVFoundation: [0] Microphone\n[1] Webcam"
                ),
                "expected": "AVFoundation: [0] Microphone\n[1] Webcam"
            }
        ),
        (
            "failed_device_listing",
            {
                "side_effect": subprocess.CalledProcessError(1, ["ffmpeg"], stderr="No devices found"),
                "raises": SystemExit
            }
        )
    ]

    def setup_method(self):
        logger.handlers = []  # Clear handlers to avoid duplicate logs

    @pytest.mark.parametrize("test_name,case", test_cases)
    def test_list_avfoundation_devices(self, test_name, case):
        """Test listing avfoundation devices with various outcomes."""
        # Given: A mocked subprocess call with specific output or error
        with patch("subprocess.run") as mock_run:
            if "side_effect" in case:
                mock_run.side_effect = case["side_effect"]
            else:
                mock_run.return_value = case["return_value"]
            # When: The function is called
            if "raises" in case:
                # Then: It raises the expected exception
                with pytest.raises(case["raises"]):
                    list_avfoundation_devices()
            else:
                # Then: The output matches the expected device list
                result = list_avfoundation_devices()
                expected = case["expected"]
                assert result == expected, f"Expected output {expected}, got {result}"

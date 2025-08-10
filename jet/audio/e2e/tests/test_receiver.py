# jet_python_modules/jet/audio/e2e/test_receiver.py
import pytest
from jet.audio.e2e.receiver import get_receiver_command


class TestReceiverCommand:
    def test_get_receiver_command_with_defaults(self):
        # Given default parameters for receiver command
        sdp_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"
        output_pattern = "last5min_%Y%m%d-%H%M%S.wav"

        # When retrieving the receiver command
        result = get_receiver_command(sdp_file, output_pattern)

        # Then the command matches the expected ffmpeg arguments for RTP audio receiving and segmenting
        expected = [
            "ffmpeg",
            "-protocol_whitelist", "file,rtp,udp",
            "-i", "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "segment",
            "-segment_time", "300",
            "-reset_timestamps", "1",
            "-strftime", "1",
            "last5min_%Y%m%d-%H%M%S.wav"
        ]
        assert result == expected

    def test_get_receiver_command_with_custom_values(self):
        # Given custom parameters for receiver command
        sdp_file = "custom.sdp"
        output_pattern = "output_%Y%m%d.wav"

        # When retrieving the receiver command
        result = get_receiver_command(sdp_file, output_pattern)

        # Then the command matches the expected ffmpeg arguments with custom values
        expected = [
            "ffmpeg",
            "-protocol_whitelist", "file,rtp,udp",
            "-i", "custom.sdp",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "segment",
            "-segment_time", "300",
            "-reset_timestamps", "1",
            "-strftime", "1",
            "output_%Y%m%d.wav"
        ]
        assert result == expected

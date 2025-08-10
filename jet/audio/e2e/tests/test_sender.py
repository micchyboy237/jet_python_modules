import pytest
from jet.audio.e2e.sender import get_sender_command


class TestSenderCommand:
    def test_get_sender_command_with_defaults(self):
        # Given default parameters for sender command
        ip = "192.168.68.104"
        port = 5004
        sdp_file = "stream.sdp"

        # When retrieving the sender command
        result = get_sender_command(ip, port, sdp_file)

        # Then the command matches the expected ffmpeg arguments for RTP audio sending
        expected = [
            "ffmpeg",
            "-loglevel", "debug",
            "-f", "avfoundation",
            "-i", ":0",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "rtp",
            "rtp://192.168.68.104:5004",
            "-sdp_file", "stream.sdp"
        ]
        assert result == expected

    def test_get_sender_command_with_custom_values(self):
        # Given custom parameters for sender command
        ip = "127.0.0.1"
        port = 1234
        sdp_file = "custom.sdp"

        # When retrieving the sender command
        result = get_sender_command(ip, port, sdp_file)

        # Then the command matches the expected ffmpeg arguments with custom values
        expected = [
            "ffmpeg",
            "-loglevel", "debug",
            "-f", "avfoundation",
            "-i", ":0",
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "rtp",
            "rtp://127.0.0.1:1234",
            "-sdp_file", "custom.sdp"
        ]
        assert result == expected

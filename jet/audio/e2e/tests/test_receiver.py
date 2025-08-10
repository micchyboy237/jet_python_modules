import pytest
from jet.audio.e2e.receiver import get_receiver_command


class TestReceiverCommand:
    def test_get_receiver_command_with_defaults(self):
        # Given default parameters for receiver command
        sdp_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"
        output_file = "last5min.wav"
        insights_file = "audio_insights.log"

        # When retrieving the receiver command
        result = get_receiver_command(sdp_file, output_file, insights_file)

        # Then the command matches the expected ffmpeg arguments for RTP audio receiving
        expected = [
            "ffmpeg",
            "-loglevel", "debug",
            "-protocol_whitelist", "file,rtp,udp",
            "-rtbufsize", "100M",
            "-i", sdp_file,
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-t", "300",
            "-y",
            "-filter:a", "volumedetect",
            "-f", "wav",
            output_file,
            "-f", "null",
            "-",
            "-report"
        ]
        assert result == expected

    def test_get_receiver_command_with_custom_values(self):
        # Given custom parameters for receiver command
        sdp_file = "custom.sdp"
        output_file = "custom_output.wav"
        insights_file = "custom_insights.log"

        # When retrieving the receiver command
        result = get_receiver_command(sdp_file, output_file, insights_file)

        # Then the command matches the expected ffmpeg arguments with custom values
        expected = [
            "ffmpeg",
            "-loglevel", "debug",
            "-protocol_whitelist", "file,rtp,udp",
            "-rtbufsize", "100M",
            "-i", sdp_file,
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-t", "300",
            "-y",
            "-filter:a", "volumedetect",
            "-f", "wav",
            output_file,
            "-f", "null",
            "-",
            "-report"
        ]
        assert result == expected

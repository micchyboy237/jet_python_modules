# jet_python_modules/jet/audio/e2e/receiver.py
import subprocess
from typing import List


def get_receiver_command(sdp_file: str, output_pattern: str) -> List[str]:
    return [
        "ffmpeg",
        "-protocol_whitelist", "file,rtp,udp",
        "-i", sdp_file,
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "segment",
        "-segment_time", "300",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_pattern
    ]


def run_receiver(
    sdp_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp",
    output_pattern: str = "last5min_%Y%m%d-%H%M%S.wav"
):
    cmd = get_receiver_command(sdp_file, output_pattern)
    subprocess.Popen(cmd)


if __name__ == "__main__":
    run_receiver()

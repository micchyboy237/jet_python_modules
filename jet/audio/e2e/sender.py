# jet_python_modules/jet/audio/e2e/sender.py
import subprocess
from typing import List


def get_sender_command(ip: str, port: int, sdp_file: str) -> List[str]:
    return [
        "ffmpeg",
        "-f", "avfoundation",
        "-i", ":0",
        "-acodec", "pcm_s16be",
        "-ar", "48000",
        "-ac", "2",
        "-f", "rtp",
        f"rtp://{ip}:{port}",
        "-sdp_file", sdp_file
    ]


def run_sender(ip: str = "192.168.68.104", port: int = 5004, sdp_file: str = "stream.sdp"):
    cmd = get_sender_command(ip, port, sdp_file)
    subprocess.Popen(cmd)


if __name__ == "__main__":
    run_sender()

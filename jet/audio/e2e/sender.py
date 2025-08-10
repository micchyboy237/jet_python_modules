import subprocess
import logging
from typing import List

logging.basicConfig(
    filename='sender.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_sender_command(ip: str, port: int, sdp_file: str) -> List[str]:
    return [
        "ffmpeg",
        "-loglevel", "debug",
        "-f", "avfoundation",
        "-i", ":0",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "rtp",
        f"rtp://{ip}:{port}",
        "-sdp_file", sdp_file
    ]


def run_sender(ip: str = "192.168.68.104", port: int = 5004, sdp_file: str = "stream.sdp"):
    try:
        cmd = get_sender_command(ip, port, sdp_file)
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
    except Exception as e:
        logging.error(f"Sender failed: {e}")
        raise


if __name__ == "__main__":
    # Debug: List available input devices
    subprocess.run(["ffmpeg", "-f", "avfoundation",
                   "-list_devices", "true", "-i", ""], stderr=subprocess.PIPE)
    run_sender()

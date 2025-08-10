import subprocess
import logging
from typing import List

# Configure logging to write to a file for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sender_debug.log"),
        logging.StreamHandler()
    ]
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
        logging.debug(f"Executing sender command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,  # Capture stdout for additional debug info
            universal_newlines=True
        )
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        logging.debug(f"FFmpeg stdout: {stdout}")
        logging.debug(f"FFmpeg stderr: {stderr}")
        if process.returncode != 0:
            logging.error(
                f"FFmpeg process failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
        logging.debug("Sender process completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Sender subprocess error: {e}, stderr: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Sender failed: {e}")
        raise


if __name__ == "__main__":
    # Debug: List available input devices
    subprocess.run(["ffmpeg", "-f", "avfoundation",
                   "-list_devices", "true", "-i", ""], stderr=subprocess.PIPE)
    run_sender()

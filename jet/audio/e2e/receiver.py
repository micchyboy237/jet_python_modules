from codecs import ignore_errors
import os
import subprocess
import logging
from typing import List

BASE_DIR = os.path.expanduser("~/generated/audio")
AUDIO_DIR = f"{BASE_DIR}/sounds"
LOGS_DIR = f"{BASE_DIR}/logs"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging to write to a file for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOGS_DIR}/receiver_debug.log"),
        logging.StreamHandler()
    ]
)


def get_receiver_command(sdp_file: str, output_file: str, insights_file: str) -> List[str]:
    return [
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


def run_receiver(
    sdp_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp",
    output_file: str = f"{AUDIO_DIR}/last5min.wav",
    insights_file: str = f"{LOGS_DIR}/audio_insights.log"
):
    try:
        cmd = get_receiver_command(sdp_file, output_file, insights_file)
        logging.debug(f"Executing receiver command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,  # Capture stdout for additional debug info
            universal_newlines=True
        )
        # Wait for the process to complete and capture output
        stdout, stderr = process.communicate()
        logging.debug(f"FFmpeg stdout: {stdout}")
        logging.debug(f"FFmpeg stderr: {stderr}")
        # Log volumedetect output to insights_file
        for line in stderr.splitlines():
            if "volumedetect" in line:
                logging.debug(
                    f"Writing volumedetect line to insights file: {line}")
                with open(insights_file, "a") as f:
                    f.write(line + "\n")
        if process.returncode != 0:
            logging.error(
                f"FFmpeg process failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
        logging.debug("Receiver process completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Receiver subprocess error: {e}, stderr: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Receiver failed: {e}")
        raise


if __name__ == "__main__":
    run_receiver()

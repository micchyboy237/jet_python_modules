from codecs import ignore_errors
import os
import subprocess
import logging
from typing import List

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.expanduser("~/generated/audio")
AUDIO_DIR = f"{BASE_DIR}/sounds"
LOGS_DIR = f"{BASE_DIR}/logs"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def get_receiver_command(output_file: str, insights_file: str) -> List[str]:
    # Ensure output_file is in AUDIO_DIR
    output_path = os.path.join(AUDIO_DIR, os.path.basename(output_file))
    return [
        "ffmpeg",
        "-loglevel", "debug",
        "-protocol_whitelist", "file,rtp,udp",
        "-f", "rtp",
        "-i", "rtp://0.0.0.0:5004",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "segment",
        "-segment_time", "300",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_path,
        "-filter:a", "volumedetect",
        "-f", "null",
        "-"
    ]


def run_receiver(
    output_file: str = "last5min_%Y%m%d-%H%M%S.wav",
    insights_file: str = "audio_insights.log"
):
    try:
        # Ensure insights_file is in LOGS_DIR
        insights_path = os.path.join(LOGS_DIR, os.path.basename(insights_file))
        cmd = get_receiver_command(output_file, insights_path)
        logger.debug(f"Executing receiver command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        logger.debug(f"FFmpeg stdout: {stdout}")
        logger.debug(f"FFmpeg stderr: {stderr}")
        for line in stderr.splitlines():
            if "volumedetect" in line:
                logger.debug(
                    f"Writing volumedetect line to insights file: {line}")
                with open(insights_path, "a") as f:
                    f.write(line + "\n")
        if process.returncode != 0:
            logger.error(
                f"Receiver process failed with return code {process.returncode}: {stderr}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
        logger.debug("Receiver process completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Receiver process error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in receiver: {str(e)}")
        raise


if __name__ == "__main__":
    run_receiver()

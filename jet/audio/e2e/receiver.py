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


def get_receiver_command(sdp_file: str, output_file: str, insights_file: str) -> List[str]:
    # Ensure output_file is in AUDIO_DIR
    output_path = os.path.join(AUDIO_DIR, os.path.basename(output_file))
    return [
        "ffmpeg",
        "-loglevel", "debug",
        "-protocol_whitelist", "file,rtp,udp",
        "-i", sdp_file,
        # Main audio output: high-quality WAV with 24-bit depth, 48kHz, mono, and filters
        "-acodec", "pcm_s24le",
        "-ar", "48000",
        "-ac", "1",
        "-af", "afftdn=nf=-25,highpass=f=200,lowpass=f=3000,loudnorm=I=-23:LRA=11:tp=-2",
        "-f", "segment",
        "-segment_time", "300",
        "-reset_timestamps", "1",
        "-strftime", "1",
        output_path,
        # Separate stream for volumedetect to avoid affecting main audio
        "-map", "0:a",
        "-filter:a", "volumedetect",
        "-f", "null",
        "-"
    ]


def run_receiver(
    sdp_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp",
    output_file: str = "last5min_%Y%m%d-%H%M%S.wav",
    insights_file: str = "audio_insights.log"
):
    try:
        # Ensure insights_file is in LOGS_DIR
        insights_path = os.path.join(LOGS_DIR, os.path.basename(insights_file))
        cmd = get_receiver_command(sdp_file, output_file, insights_path)
        logger.debug(f"Executing receiver command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        # Log output in real-time
        while True:
            try:
                stderr_line = process.stderr.readline()
                if stderr_line:
                    logger.debug(f"FFmpeg stderr: {stderr_line.strip()}")
                    if "volumedetect" in stderr_line.lower():
                        logger.debug(
                            f"Writing volumedetect line to insights file: {stderr_line.strip()}")
                        with open(insights_path, "a") as f:
                            f.write(stderr_line + "\n")
                stdout_line = process.stdout.readline()
                if stdout_line:
                    logger.debug(f"FFmpeg stdout: {stdout_line.strip()}")
                if process.poll() is not None:
                    break
            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping receiver")
                process.terminate()
                process.wait()
                break
        stdout, stderr = process.communicate()
        logger.debug(f"FFmpeg stdout: {stdout}")
        logger.debug(f"FFmpeg stderr: {stderr}")
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

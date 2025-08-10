import subprocess
from typing import List

from jet.logger import logger


def get_sender_command(ip: str, port: int, sdp_file: str) -> List[str]:
    return [
        "ffmpeg",
        "-loglevel", "debug",
        "-f", "avfoundation",
        "-i", ":0",  # Updated to use default audio device index 0
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-f", "rtp",
        f"rtp://{ip}:{port}",
        "-sdp_file", sdp_file
    ]


def check_audio_device() -> bool:
    """Check if an audio input device is available for avfoundation."""
    try:
        cmd = ["ffmpeg", "-f", "avfoundation",
               "-list_devices", "true", "-i", "dummy"]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        stderr_output = result.stderr.lower()
        if "avfoundation audio devices" in stderr_output and "[avfoundation @" in stderr_output:
            logger.debug("Audio devices found in FFmpeg output")
            return True
        logger.error("No audio devices detected in FFmpeg output")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list audio devices: {e.stderr}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error while checking audio devices: {str(e)}")
        return False


def run_sender(ip: str = "192.168.68.104", port: int = 5004, sdp_file: str = "stream.sdp", timeout: int = 10):
    if not check_audio_device():
        raise RuntimeError("No audio input device available")
    try:
        cmd = get_sender_command(ip, port, sdp_file)
        logger.debug(f"Running sender with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode != 0:
            logger.error(
                f"Sender process failed with return code {process.returncode}: {stderr}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        logger.error(f"Sender process timed out: {stderr}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Sender process error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in sender: {str(e)}")
        raise


if __name__ == "__main__":
    subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "dummy"],
        stderr=subprocess.PIPE
    )
    run_sender()

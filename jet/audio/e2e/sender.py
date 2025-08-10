import subprocess
from typing import List

from jet.logger import logger

SDP_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"


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
    """Check if an audio input device is available for avfoundation by testing the default input."""
    try:
        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-i", ":0",  # Test the default audio device
            "-t", "1",  # Capture for 1 second to minimize overhead
            "-f", "null",  # Output to null to avoid creating files
            "-"
        ]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=False  # Allow non-zero exit codes for robust checking
        )
        stderr_output = result.stderr.lower()
        logger.debug(f"FFmpeg audio device check output: {stderr_output}")
        if result.returncode == 0:
            logger.debug("Audio device ':0' is available and functional")
            return True
        if "input/output error" in stderr_output or "no such file" in stderr_output:
            logger.error(
                "Audio device ':0' is not available or not functional")
        else:
            logger.error(
                f"Unexpected FFmpeg error while checking audio device: {stderr_output}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while checking audio device: {str(e)}")
        return False


def run_sender(ip: str = "192.168.68.104", port: int = 5004, sdp_file: str = SDP_FILE):
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
        # Log output in real-time without timeout
        while True:
            try:
                stderr_line = process.stderr.readline()
                if stderr_line:
                    logger.debug(f"FFmpeg stderr: {stderr_line.strip()}")
                stdout_line = process.stdout.readline()
                if stdout_line:
                    logger.debug(f"FFmpeg stdout: {stdout_line.strip()}")
                if process.poll() is not None:
                    break
            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping sender")
                process.terminate()
                process.wait()
                break
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(
                f"Sender process failed with return code {process.returncode}: {stderr}")
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stderr=stderr)
        logger.debug("Sender process completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Sender process error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in sender: {str(e)}")
        raise


if __name__ == "__main__":
    subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true"],
        stderr=subprocess.PIPE
    )
    run_sender()

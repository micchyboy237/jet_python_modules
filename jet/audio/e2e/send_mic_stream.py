import sys
import subprocess
import socket
import time
import signal
import argparse
from typing import Optional
from jet.audio.record_mic_stream import list_avfoundation_devices
from jet.audio.utils import capture_and_save_audio
from jet.logger import logger


def stream_audio_to_receiver(
    receiver_ip: str, port: int, sample_rate: int, channels: int, device_index: str
) -> subprocess.Popen:
    """Stream audio to a receiver via RTP."""
    try:
        socket.gethostbyname(receiver_ip)
    except socket.error as e:
        logger.error(f"Invalid receiver IP: {e}")
        sys.exit(1)
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-report",
        "-f", "avfoundation", "-i", f"none:{device_index}",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "rtp",
        f"rtp://{receiver_ip}:{port}?rtcpport={port}&pkt_size=188&payload_type=11&buffer_size=1000000"
    ]
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
        )
    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)
    return process


def send_mic_stream(
    receiver_ip: Optional[str] = None,
    port: Optional[int] = None,
    sample_rate: int = 44100,
    channels: int = 2,
    file_prefix: str = "recording",
    device_index: str = "1"
):
    """Orchestrate audio capture, saving, and optional streaming to a receiver."""
    try:
        subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-i",
                f"none:{device_index}", "-t", "1", "-f", "null", "-"],
            capture_output=True, text=True, timeout=5, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Invalid device index {device_index}: {e.stderr}")
        logger.info(f"Available devices:\n{list_avfoundation_devices()}")
        logger.info(
            "Ensure FFmpeg has microphone access in System Settings > Privacy & Security > Microphone.")
        sys.exit(1)
    processes = []
    if receiver_ip and port:
        logger.info(f"Starting audio streaming to {receiver_ip}:{port}")
        processes.append(stream_audio_to_receiver(
            receiver_ip, port, sample_rate, channels, device_index))
    else:
        logger.info(
            "No receiver IP/port provided, only capturing and saving audio locally")
    logger.info(f"Starting audio capture and saving to {file_prefix}_*.wav")
    processes.append(capture_and_save_audio(
        sample_rate, channels, file_prefix, device_index
    ))
    try:
        start_time = time.time()
        min_runtime = 60
        while time.time() - start_time < min_runtime:
            for process in processes:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(
                        f"FFmpeg process {process.pid} exited unexpectedly: {stderr}")
                    sys.exit(1)
                line = process.stderr.readline()
                if line:
                    logger.debug(line.strip())
            time.sleep(0.1)
        for process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"FFmpeg process {process.pid} failed: {stderr}")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Terminating FFmpeg processes...")
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"FFmpeg process {process.pid} did not terminate gracefully, killing...")
                process.kill()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send microphone audio via RTP and save a WAV file.")
    parser.add_argument("--receiver-ip", type=str, default=None,
                        help="IP address of the RTP receiver (optional)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port number for RTP streaming (optional)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Audio sample rate (Hz, default: 44100)")
    parser.add_argument("--channels", type=int, default=2, choices=[
                        1, 2], help="Number of audio channels (1=mono, 2=stereo, default: 2)")
    parser.add_argument("--file-prefix", type=str, default="recording",
                        help="Prefix for output WAV file (default: recording)")
    parser.add_argument("--device-index", type=str, default="1",
                        help="avfoundation device index for microphone (default: 1)")
    args = parser.parse_args()
    if (args.receiver_ip is None) != (args.port is None):
        parser.error(
            "Both --receiver-ip and --port must be provided together or both omitted")
    send_mic_stream(
        args.receiver_ip,
        args.port,
        args.sample_rate,
        args.channels,
        args.file_prefix,
        args.device_index
    )

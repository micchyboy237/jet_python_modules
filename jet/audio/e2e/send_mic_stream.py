import sys
import subprocess
import socket
import logging
import time
import signal
import argparse
from typing import Optional


def capture_and_save_audio(sample_rate: int, channels: int, segment_time: int, file_prefix: str) -> subprocess.Popen:
    """Capture audio from microphone and save to segmented WAV files."""
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-report",
        "-f", "avfoundation", "-i", "none:1",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "segment",
        "-segment_time", str(segment_time), "-segment_format", "wav",
        f"-segment_list={file_prefix}_list.txt",
        f"{file_prefix}_%05d.wav"
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )
    return process


def stream_audio_to_receiver(receiver_ip: str, port: int, sample_rate: int, channels: int) -> subprocess.Popen:
    """Stream audio to a receiver via RTP."""
    try:
        socket.gethostbyname(receiver_ip)
    except socket.error as e:
        logging.error(f"Invalid receiver IP: {e}")
        sys.exit(1)

    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-report",
        "-f", "avfoundation", "-i", "none:1",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "rtp",
        f"rtp://{receiver_ip}:{port}?rtcpport={port}&pkt_size=188&payload_type=11&buffer_size=1000000"
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )
    return process


def send_mic_stream(
    receiver_ip: Optional[str] = None,
    port: Optional[int] = None,
    sample_rate: int = 44100,
    channels: int = 2,
    segment_time: int = 30,
    file_prefix: str = "recording"
):
    """Orchestrate audio capture, saving, and optional streaming to a receiver."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    processes = []
    if receiver_ip and port:
        logging.info(f"Starting audio streaming to {receiver_ip}:{port}")
        processes.append(stream_audio_to_receiver(
            receiver_ip, port, sample_rate, channels))
    else:
        logging.info(
            "No receiver IP/port provided, only capturing and saving audio locally")

    logging.info(f"Starting audio capture and saving to {file_prefix}_*.wav")
    processes.append(capture_and_save_audio(
        sample_rate, channels, segment_time, file_prefix))

    def signal_handler(sig, frame):
        logging.info("Terminating FFmpeg processes...")
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning(
                    f"FFmpeg process {process.pid} did not terminate gracefully, killing...")
                process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        start_time = time.time()
        min_runtime = 60
        while time.time() - start_time < min_runtime:
            for process in processes:
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logging.error(
                        f"FFmpeg process {process.pid} exited unexpectedly: {stderr}")
                    sys.exit(1)
                line = process.stderr.readline()
                if line:
                    logging.debug(line.strip())
            time.sleep(0.1)
        for process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logging.error(f"FFmpeg process {process.pid} failed: {stderr}")
                sys.exit(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

import sys
import subprocess
import socket
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


def send_mic_stream(receiver_ip: str, port: int = 5000):
    """Send mic audio to a receiver via RTP and save a local copy."""
    try:
        socket.gethostbyname(receiver_ip)
    except socket.error as e:
        logging.error(f"Invalid receiver IP: {receiver_ip} - {e}")
        sys.exit(1)
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-f", "avfoundation", "-i", "none:1",
        "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "tee",
        f"[f=rtp]rtp://{receiver_ip}:{port}?rtcpport={port}|[f=wav]recording.wav"
    ]
    logging.info(
        f"Sending mic audio to {receiver_ip}:{port} and saving as recording.wav")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    logging.debug(f"FFmpeg output: {stderr}")
    if process.returncode != 0:
        logging.error(f"FFmpeg failed with exit code {process.returncode}")
        sys.exit(1)

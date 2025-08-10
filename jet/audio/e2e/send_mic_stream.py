import sys
import subprocess
import socket


def send_mic_stream(receiver_ip: str, port: int = 5000):
    """Send mic audio to a receiver via RTP and save a local copy."""
    try:
        socket.gethostbyname(receiver_ip)
    except socket.error as e:
        sys.exit(f"Invalid receiver IP: {receiver_ip} - {e}")

    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-f", "avfoundation", "-i", "none:1",
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16be",
        "-map", "0:a",
        "-f", "tee",
        f"[f=rtp]rtp://{receiver_ip}:{port}?rtcpport={port}|[f=wav:c=a:pcm_s16le]recording.wav"
    ]

    print(
        f"Sending mic audio to {receiver_ip}:{port} and saving as recording.wav")
    subprocess.run(cmd)

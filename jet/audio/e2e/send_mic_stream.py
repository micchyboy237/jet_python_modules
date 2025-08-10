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
        # Map for RTP output
        "-map", "0:a", "-c:a:0", "pcm_s16be", "-ar:0", "44100", "-ac:0", "2",
        # Map for WAV output
        "-map", "0:a", "-c:a:1", "pcm_s16le", "-ar:1", "44100", "-ac:1", "2",
        "-f", "tee",
        f"[select=a:0:f=rtp]rtp://{receiver_ip}:{port}?rtcpport={port}|[select=a:1:f=wav]recording.wav"
    ]

    print(
        f"Sending mic audio to {receiver_ip}:{port} and saving as recording.wav")
    subprocess.run(cmd)

import sys
import subprocess
import socket
import logging
import time
import signal


def send_mic_stream(receiver_ip: str, port: int = 5000):
    """Send mic audio to a receiver via RTP and save a local copy."""
    try:
        socket.gethostbyname(receiver_ip)
    except socket.error as e:
        sys.exit(1)
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets",
        "-report",
        "-f", "avfoundation", "-i", "none:1",
        "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "rtp",
        f"rtp://{receiver_ip}:{port}?rtcpport={port+1}&pkt_size=188&payload_type=11&buffer_size=1000000",
        "-f", "wav", "recording.wav"
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    def signal_handler(sig, frame):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    try:
        start_time = time.time()
        min_runtime = 60
        while time.time() - start_time < min_runtime:
            if process.poll() is not None:
                stdout, stderr = process.comready()
                sys.exit(1)
            line = process.stderr.readline()
            if line:
                time.sleep(0.1)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            sys.exit(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

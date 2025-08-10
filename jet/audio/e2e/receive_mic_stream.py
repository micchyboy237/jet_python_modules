import sys
import socket
import subprocess
from pathlib import Path
import signal
import time
import logging


def get_local_ip() -> str:
    """Get local IP address used for outbound traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except socket.error as e:
        raise
    finally:
        s.close()
    return ip


def generate_sdp(ip: str, port: int, filename: Path):
    """Create an SDP file for receiving PCM S16LE audio."""
    sdp_content = f"""v=0
o=- 0 0 IN IP4 {ip}
s=Audio Stream
c=IN IP4 {ip}
t=0 0
m=audio {port} RTP/AVP 11
a=rtpmap:11 L16/48000/2
a=control:streamid=0
a=recvonly
"""
    filename.write_text(sdp_content)


def receive_stream(port: int = 5000, output_wav: str = "output.wav"):
    ip = get_local_ip()
    sdp_file = Path("stream.sdp")
    generate_sdp(ip, port, sdp_file)
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "debug",
        "-report",
        "-protocol_whitelist", "file,udp,rtp",
        "-c:a", "pcm_s16le",
        "-i", str(sdp_file),
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
        "-avioflags", "direct",
        "-timeout", "30000000",
        "-buffer_size", "2000000",
        "-f", "wav",
        output_wav
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
                stdout, stderr = process.communicate()
                sys.exit(1)
            line = process.stderr.readline()
            if line:
                time.sleep(0.1)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            sys.exit(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

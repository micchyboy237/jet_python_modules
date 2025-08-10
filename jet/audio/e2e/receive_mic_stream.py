import sys
import socket
import subprocess
from pathlib import Path


def get_local_ip() -> str:
    """Get local IP address used for outbound traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def generate_sdp(ip: str, port: int, filename: Path):
    """Create an SDP file for receiving PCM S16BE audio."""
    sdp_content = f"""v=0
o=- 0 0 IN IP4 {ip}
s=Audio Stream
c=IN IP4 {ip}
t=0 0
m=audio {port} RTP/AVP 11
a=rtpmap:11 L16/44100/2
a=fmtp:11
a=control:streamid=0
a=buffer_size:1000000
a=recvonly
"""
    filename.write_text(sdp_content)
    print(f"Generated SDP file at {filename}")


def receive_stream(port: int = 5000, output_wav: str = "output.wav"):
    ip = get_local_ip()
    sdp_file = Path("stream.sdp")
    generate_sdp(ip, port, sdp_file)

    cmd = [
        "ffmpeg", "-loglevel", "debug",
        "-protocol_whitelist", "file,udp,rtp",
        "-i", str(sdp_file),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        output_wav
    ]

    print(f"Listening on {ip}:{port}, saving audio to {output_wav}")
    subprocess.run(cmd)

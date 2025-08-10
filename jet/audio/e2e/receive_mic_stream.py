import sys
import socket
import subprocess
from pathlib import Path
import signal
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


def get_local_ip() -> str:
    """Get local IP address used for outbound traffic."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except socket.error as e:
        logging.error(f"Failed to get local IP: {e}")
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
a=rtpmap:11 PCM/44100/2
a=fmtp:11 bitorder=little
a=control:streamid=0
a=recvonly
"""
    filename.write_text(sdp_content)
    logging.info(f"Generated SDP file at {filename}")


def receive_stream(port: int = 5000, output_wav: str = "output.wav"):
    ip = get_local_ip()
    sdp_file = Path("stream.sdp")
    generate_sdp(ip, port, sdp_file)
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "debug",
        "-protocol_whitelist", "file,udp,rtp",
        "-i", str(sdp_file),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-f", "wav",
        output_wav
    ]
    logging.info(f"Listening on {ip}:{port}, saving audio to {output_wav}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )

    def signal_handler(sig, frame):
        logging.info("Received interrupt, shutting down FFmpeg gracefully...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.warning(
                "FFmpeg did not terminate in time, forcing shutdown...")
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        # Log FFmpeg output in real-time
        start_time = time.time()
        min_runtime = 60  # Increased to 60 seconds for debugging
        while time.time() - start_time < min_runtime:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logging.error(
                    f"FFmpeg exited unexpectedly. Stdout: {stdout}, Stderr: {stderr}")
                sys.exit(1)
            # Read and log FFmpeg stderr line by line
            line = process.stderr.readline()
            if line:
                logging.debug(f"FFmpeg: {line.strip()}")
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        logging.debug(f"FFmpeg final output: {stderr}")
        if process.returncode != 0:
            logging.error(f"FFmpeg failed with exit code {process.returncode}")
            sys.exit(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

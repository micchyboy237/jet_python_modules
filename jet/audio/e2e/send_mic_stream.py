import sys
import subprocess
import socket
import logging
import time
import signal

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
        "-f", "rtp",
        f"rtp://{receiver_ip}:{port}?rtcpport={port}&pkt_size=188&payload_type=96",
        "-f", "wav", "recording.wav"
    ]
    logging.info(
        f"Sending mic audio to {receiver_ip}:{port} and saving as recording.wav")
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
        min_runtime = 15  # seconds
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

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import time

SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_LISTEN_IP = "0.0.0.0"
DEFAULT_PORT = "5000"
DEFAULT_SDP_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"


def receive_mic_stream(
    output_file: Path,
    listen_ip: str = DEFAULT_LISTEN_IP,
    port: str = DEFAULT_PORT,
    stream_sdp_path: str = DEFAULT_SDP_FILE
) -> Optional[subprocess.Popen]:
    """
    Receive audio stream over RTP and save to a WAV file.
    Args:
        output_file: Path to save the output WAV file
        listen_ip: IP address to listen on (default: "0.0.0.0" for all interfaces)
        port: Port to listen for the RTP stream
        stream_sdp_path: Path to the SDP file describing the incoming RTP stream
    Returns:
        subprocess.Popen object if receiving started successfully, None otherwise
    """
    try:
        sdp_file = Path(stream_sdp_path)
        if not sdp_file.exists():
            print(f"❌ Error: SDP file {sdp_file} not found. Please create it.")
            return None
        print(f"📄 Using SDP file: {sdp_file}")
        # Log SDP file contents for debugging
        with sdp_file.open('r') as f:
            print(f"DEBUG: SDP file contents:\n{f.read()}")
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "debug",
            "-y",
            "-protocol_whitelist", "file,udp,rtp",
            "-i", f"rtp://{listen_ip}:{port}?localaddr={listen_ip}",
            "-f", "sdp",
            "-i", f"file://{sdp_file}",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le",
            str(output_file),
        ]
        print(
            f"🎧 Receiving stream on {listen_ip}:{port} and saving to {output_file}...")
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        def log_packets():
            packet_count = 0
            max_packets_to_log = 5
            while process.poll() is None:
                line = process.stderr.readline()
                if not line:
                    continue
                if "RTP: packet received" in line or "Received packet" in line:
                    packet_count += 1
                    if packet_count <= max_packets_to_log:
                        print(
                            f"📡 Sound received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    elif packet_count == max_packets_to_log + 1:
                        print(
                            "📡 Further packet receives suppressed to avoid flooding logs")
                print(f"DEBUG: FFmpeg: {line.strip()}")

        def log_status():
            while process.poll() is None:
                print(
                    f"📡 Receiving active at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(5)
        threading.Thread(target=log_packets, daemon=True).start()
        threading.Thread(target=log_status, daemon=True).start()
        return process
    except FileNotFoundError:
        print("❌ Error: FFmpeg is not installed or not found in PATH")
        return None
    except Exception as e:
        print(f"❌ Error starting receiver: {str(e)}")
        return None

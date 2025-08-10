import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_LISTEN_IP = "0.0.0.0"  # Listen on all interfaces
DEFAULT_PORT = "5000"


def receive_mic_stream(output_file: Path, listen_ip: str = DEFAULT_LISTEN_IP, port: str = DEFAULT_PORT) -> Optional[subprocess.Popen]:
    """
    Receive audio stream over RTP and save to a WAV file.

    Args:
        output_file: Path to save the output WAV file
        listen_ip: IP address to listen on (default: "0.0.0.0" for all interfaces)
        port: Port to listen for the RTP stream

    Returns:
        subprocess.Popen object if receiving started successfully, None otherwise
    """
    try:
        # Construct FFmpeg command for receiving RTP stream
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "debug",  # Enable debug logging
            "-y",  # Overwrite output file if it exists
            "-protocol_whitelist", "file,udp,rtp",
            "-f", "rtp",
            "-i", f"rtp://{listen_ip}:{port}",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le",  # 16-bit PCM for compatibility
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

        # Thread to monitor FFmpeg stderr for packet receiving
        def log_packets():
            packet_count = 0
            max_packets_to_log = 5  # Limit to avoid flooding
            while process.poll() is None:
                line = process.stderr.readline()
                if not line:
                    continue
                if "Received packet from" in line or "RTP: packet received" in line:
                    packet_count += 1
                    if packet_count <= max_packets_to_log:
                        print(
                            f"📡 Sound received at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    elif packet_count == max_packets_to_log + 1:
                        print(
                            "📡 Further packet receives suppressed to avoid flooding logs")
                print(f"DEBUG: FFmpeg: {line.strip()}")

        # Thread for periodic receiving status
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

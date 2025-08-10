import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import time

SAMPLE_RATE = 44100
CHANNELS = 2
DEFAULT_DEST_IP = "127.0.0.1"
DEFAULT_PORT = "5000"
DEFAULT_SDP_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/e2e/stream.sdp"

def get_ffmpeg_input_device() -> str:
    return "avfoundation"

def list_avfoundation_devices():
    # Placeholder: Implement actual device listing logic if needed
    return [], ['BlackHole 2ch', 'MacBook Air Microphone', 'Microsoft Teams Audio']

def send_mic_stream(
    duration: int,
    dest_ip: str = DEFAULT_DEST_IP,
    port: str = DEFAULT_PORT,
    audio_index: str = "1",
    stream_sdp_path: str = DEFAULT_SDP_FILE,
    output_file: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """
    Stream audio from microphone to a remote receiver using FFmpeg over RTP and optionally save to a file.
    Args:
        duration: Duration to stream in seconds (0 for indefinite)
        dest_ip: Destination IP address of the receiver
        port: Destination port for the RTP stream
        audio_index: Audio device index for avfoundation (default: "1" for MacBook Air Microphone)
        stream_sdp_path: Path to the SDP file to use for the receiver configuration (not modified)
        output_file: Path to save the recorded audio (e.g., 'output.wav'), None if not saving
    Returns:
        subprocess.Popen object if streaming started successfully, None otherwise
    """
    try:
        input_device = get_ffmpeg_input_device()
        _, audio_devices = list_avfoundation_devices()
        if not audio_devices:
            print("❌ No audio devices found")
            return None
        device_indices = {str(i): name for i, name in enumerate(audio_devices)}
        print(f"DEBUG: Available device indices: {device_indices}")
        selected_index = audio_index if audio_index in device_indices else "1"
        selected_device = device_indices.get(selected_index, audio_devices[0])
        print(f"DEBUG: Selected device index: {selected_index}, device name: {selected_device}")
        sdp_file = Path(stream_sdp_path)
        if not sdp_file.exists():
            print(f"❌ Error: SDP file {sdp_file} not found. Please create it.")
            return None
        print(f"📄 Receiver should use SDP file: {sdp_file}")
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "debug",
            "-re",
            "-fflags", "+flush_packets",
            "-f", input_device,
            "-i", f"none:{selected_index}",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-c:a", "pcm_s16be",
            "-map", "0:a",
            "-f", "tee",
            f"[f=rtp]rtp://{dest_ip}:{port}?rtcpport={port}\\|[f=wav:c=a:pcm_s16le]{output_file}",
        ]
        if duration > 0:
            ffmpeg_cmd.extend(["-t", str(duration)])
        print(f"DEBUG: FFmpeg command: {' '.join(ffmpeg_cmd)}")
        print(f"🎙️ Streaming ({CHANNELS} channel{'s' if CHANNELS > 1 else ''}) to {dest_ip}:{port} using device index {selected_index} ({selected_device})...")
        if output_file:
            print(f"💾 Saving audio to {output_file}")
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
                if "size=" in line and "time=" in line and "bitrate=" in line and "speed=" in line:
                    packet_count += 1
                    if packet_count <= max_packets_to_log:
                        print(f"📡 Sound sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    elif packet_count == max_packets_to_log + 1:
                        print("📡 Further packet sends suppressed to avoid flooding logs")
                print(f"DEBUG: FFmpeg: {line.strip()}")

        def log_status():
            while process.poll() is None:
                print(f"📡 Streaming active at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(5)

        threading.Thread(target=log_packets, daemon=True).start()
        threading.Thread(target=log_status, daemon=True).start()

        def signal_handler(sig, frame):
            print("🛑 Stopping sender...")
            process.terminate()
            try:
                process.wait(timeout=2)  # Allow 2 seconds for FFmpeg to finalize
            except subprocess.TimeoutExpired:
                process.kill()
            print("🛑 Streaming stopped by user")

        signal.signal(signal.SIGINT, signal_handler)
        return process

    except FileNotFoundError:
        print("❌ Error: FFmpeg is not installed or not found in PATH")
        return None
    except Exception as e:
        print(f"❌ Error starting stream: {str(e)}")
        return None
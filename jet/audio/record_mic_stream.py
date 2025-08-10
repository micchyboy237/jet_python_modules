import subprocess
from datetime import datetime
from pathlib import Path
import platform
import sys
from typing import Optional

SAMPLE_RATE = 44100
CHANNELS = 2


def get_ffmpeg_input_device() -> str:
    """Determine the appropriate FFmpeg input device based on the operating system."""
    if platform.system() == "Darwin":  # macOS
        return "avfoundation"
    elif platform.system() == "Windows":
        return "dshow"
    elif platform.system() == "Linux":
        return "alsa"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")


def record_mic_stream(duration: int, output_file: Path) -> Optional[subprocess.Popen]:
    """
    Record audio from microphone using FFmpeg and stream to a WAV file.

    Args:
        duration: Duration to record in seconds
        output_file: Path to save the output WAV file

    Returns:
        subprocess.Popen object if recording started successfully, None otherwise
    """
    try:
        input_device = get_ffmpeg_input_device()

        # Construct FFmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", input_device,
            "-i", "0" if input_device == "avfoundation" else "default",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-t", str(duration),
            "-c:a", "pcm_s16le",  # 16-bit PCM encoding
            str(output_file)
        ]

        print(
            f"🎙️ Recording ({CHANNELS} channel{'s' if CHANNELS > 1 else ''}) for {duration} seconds...")
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        return process

    except FileNotFoundError:
        print("❌ Error: FFmpeg is not installed or not found in PATH")
        return None
    except Exception as e:
        print(f"❌ Error starting recording: {str(e)}")
        return None

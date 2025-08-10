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


def list_avfoundation_devices() -> tuple[list[str], list[str]]:
    """List available AVFoundation devices (video and audio) on macOS."""
    try:
        cmd = ["ffmpeg", "-f", "avfoundation",
               "-list_devices", "true", "-i", ""]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()

        video_devices = []
        audio_devices = []
        current_section = None

        for line in stderr.splitlines():
            if "AVFoundation video devices" in line:
                current_section = "video"
            elif "AVFoundation audio devices" in line:
                current_section = "audio"
            elif current_section and line.strip().startswith("[AVFoundation"):
                device_name = line.split("]")[-1].strip()
                if current_section == "video":
                    video_devices.append(device_name)
                elif current_section == "audio":
                    audio_devices.append(device_name)

        print("🎙️ Available audio devices:", audio_devices)
        return video_devices, audio_devices

    except FileNotFoundError:
        print("❌ Error: FFmpeg is not installed or not found in PATH")
        return [], []
    except Exception as e:
        print(f"❌ Error listing devices: {str(e)}")
        return [], []


def record_mic_stream(duration: int, output_file: Path, audio_index: str = "0") -> Optional[subprocess.Popen]:
    """
    Record audio from microphone using FFmpeg and stream to a WAV file.

    Args:
        duration: Duration to record in seconds
        output_file: Path to save the output WAV file
        audio_index: Audio device index for avfoundation (default: "0")

    Returns:
        subprocess.Popen object if recording started successfully, None otherwise
    """
    try:
        input_device = get_ffmpeg_input_device()

        # List devices to help with debugging
        _, audio_devices = list_avfoundation_devices()
        if not audio_devices:
            print("❌ No audio devices found")
            return None

        # Use first audio device if available, or fall back to provided index
        selected_index = audio_index if audio_index in audio_devices else audio_devices[0].split(
            "[")[1].split("]")[0] if audio_devices else audio_index

        # Construct FFmpeg command for audio-only input
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f", input_device,
            # Explicitly specify no video, only audio
            "-i", f"none:{selected_index}",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-t", str(duration),
            "-c:a", "pcm_s16le",  # 16-bit PCM encoding
            str(output_file)
        ]

        print(
            f"🎙️ Recording ({CHANNELS} channel{'s' if CHANNELS > 1 else ''}) for {duration} seconds using device index {selected_index}...")
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

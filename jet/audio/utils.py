import sys
import subprocess
import os
import glob
import re
from typing import List, Tuple
from jet.logger import logger


def list_avfoundation_devices() -> str:
    """List available avfoundation devices and return the output."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
                "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, check=True, timeout=10
        )
        output = result.stderr
        if not output or "AVFoundation" not in output:
            logger.error(
                "Unexpected FFmpeg output format: No AVFoundation devices found.")
            sys.exit(1)
        return output
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list avfoundation devices: {e.stderr}")
        logger.info(
            "Ensure FFmpeg is installed and has microphone access in System Settings > Privacy & Security > Microphone.")
        sys.exit(1)


def get_next_file_suffix(file_prefix: str) -> int:
    """Find the next available suffix for output files to avoid overwriting."""
    pattern = os.path.join(os.getcwd(), f"{file_prefix}_[0-9]*.wav")
    existing_files = glob.glob(pattern)
    logger.debug(f"Found files matching pattern {pattern}: {existing_files}")
    suffixes = []
    for f in existing_files:
        base = os.path.basename(f)
        logger.debug(f"Processing file: {base}")
        match = re.match(rf"{file_prefix}_(\d+)\.wav$", base)
        if not match:
            logger.warning(
                f"Skipping file {base}: does not match expected pattern {file_prefix}_[number].wav")
            continue
        number_part = match.group(1)
        suffix = int(number_part.lstrip('0') or '0')
        logger.debug(f"Valid suffix found: {suffix}")
        suffixes.append(suffix)
    next_suffix = max(suffixes) + 1 if suffixes else 0
    logger.debug(f"Returning next suffix: {next_suffix}")
    return next_suffix


def capture_and_save_audio(
    sample_rate: int, channels: int, file_prefix: str, device_index: str
) -> subprocess.Popen:
    """Capture audio from microphone and save to a single WAV file."""
    start_suffix = get_next_file_suffix(file_prefix)
    output_file = f"{file_prefix}_{start_suffix:05d}.wav"
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets", "-flush_packets", "1",
        "-report",
        "-f", "avfoundation", "-i", f"none:{device_index}",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "wav",
        # "-y",  # Overwrite output file if it exists
        output_file
    ]
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
        )
        logger.info(
            f"Started FFmpeg process {process.pid} for audio capture to {output_file}")
        return process
    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)

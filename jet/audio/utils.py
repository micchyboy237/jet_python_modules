import sys
import subprocess
import os
import glob

from jet.logger import logger


def list_avfoundation_devices() -> str:
    """List available avfoundation devices and return the output."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation",
                "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, check=True, timeout=10
        )
        return result.stderr
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
        parts = base.split('_')
        if len(parts) < 2:
            logger.debug(f"Skipping file {base}: no underscore found")
            continue
        number_part = parts[-1].split('.')[0]  # e.g., '00001'
        logger.debug(f"Extracted number_part: {number_part}")
        if not number_part.isdigit():
            logger.debug(
                f"Skipping file {base}: number_part '{number_part}' is not numeric")
            continue
        suffix = int(number_part.lstrip('0') or '0')  # Handle '00000' as 0
        logger.debug(f"Valid suffix found: {suffix}")
        suffixes.append(suffix)

    next_suffix = max(suffixes) + 1 if suffixes else 0
    logger.debug(f"Returning next suffix: {next_suffix}")
    return next_suffix


def capture_and_save_audio(
    sample_rate: int, channels: int, segment_time: int, file_prefix: str,
    device_index: str, min_duration: float = 1.0, segment_flush_interval: int = 5
) -> subprocess.Popen:
    """Capture audio from microphone and save to segmented WAV files with silence trimming."""
    start_suffix = get_next_file_suffix(file_prefix)
    segment_list_file = f"{file_prefix}_list_{start_suffix:05d}.txt"
    output_pattern = f"{file_prefix}_%05d.wav"

    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets", "-flush_packets", "1",
        "-report",
        "-f", "avfoundation", "-i", f"none:{device_index}",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-af", f"silencedetect=noise=-50dB:d={min_duration},atrim",
        "-map", "0:a",
        "-f", "segment",
        "-segment_time", str(segment_time), "-segment_format", "wav",
        "-segment_list", segment_list_file, "-segment_list_type", "flat",
        "-segment_wrap", "0",
        "-segment_start_number", str(start_suffix),
        output_pattern
    ]
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
        )
    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)
    return process

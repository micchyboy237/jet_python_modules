from pathlib import Path
import sys
import subprocess
import os
import glob
import re
from typing import Iterator, List, Tuple, Union
import librosa
import sounddevice as sd
import numpy as np
from jet.logger import logger
import soundfile as sf  # <-- fast, supports many formats, free & popular

def get_input_channels() -> int:
    device_info = sd.query_devices(sd.default.device[0], 'input')
    channels = device_info['max_input_channels']
    logger.debug(f"Detected {channels} input channels")
    return channels

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
    if os.path.exists(output_file):
        logger.error(
            f"Output file {output_file} already exists. Please use a different file prefix or remove existing files.")
        sys.exit(1)
    cmd = [
        "ffmpeg", "-loglevel", "debug", "-re", "-fflags", "+flush_packets", "-flush_packets", "1",
        "-report",
        "-f", "avfoundation", "-i", f"none:{device_index}",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_s16le",
        "-map", "0:a",
        "-f", "wav",
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

def load_audio(audio_path, sample_rate: int = 16000, **kwargs) -> Tuple[np.ndarray, Union[int, float]]:
    settings = {
        "path": audio_path,
        "sr": sample_rate,
        "mono": False, # Keep stereo if present
        **kwargs
    }
    audio, sr = librosa.load(**settings)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)  # Force stereo for consistency
    return audio, sr

def split_audio(audio: Union[np.ndarray, str, Path], segment_duration: float = 20.0, overlap_duration: float = 2.0, sample_rate: int = 16000) -> Iterator[Tuple[np.ndarray, float, float]]:
    """Split audio into segments with optional overlap to prevent information loss.

    Args:
        audio: Input audio array.
        segment_duration: Duration of each segment in seconds.
        overlap_duration: Duration of overlap between segments in seconds.

    Yields:
        Tuple containing the audio segment, start time, and end time.
    """
    if isinstance(audio, (str, Path)):
        audio, sampling_rate = librosa.load(str(audio), sr=sample_rate, mono=True)

    samples_per_segment = int(sample_rate * segment_duration)
    overlap_samples = int(sample_rate * overlap_duration)
    step_samples = samples_per_segment - \
        overlap_samples  # Step size adjusted for overlap
    total_samples = len(audio)

    if step_samples <= 0:
        raise ValueError(
            "Overlap duration must be less than segment duration.")

    for start_sample in range(0, total_samples, step_samples):
        end_sample = min(start_sample + samples_per_segment, total_samples)
        segment = audio[start_sample:end_sample]
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        yield segment, start_time, end_time


def save_audio_chunks(
    chunks: Iterator[Tuple[np.ndarray, float, float]],
    output_dir: Union[str, Path],
    prefix: str = "chunk",
    format: str = "wav",          # or "flac", "ogg", etc.
    subtype: str = "PCM_16",      # good quality/default for wav
    sample_rate: int = 16000
) -> list[Path]:
    """
    Save all chunks yielded by split_audio() to individual files.
    
    Returns list of created file paths (useful for downstream processing or testing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, (segment, start_sec, end_sec) in enumerate(chunks):
        # Human-readable filename with zero-padded index and time range
        filename = f"{prefix}_{i:04d}_{start_sec:.2f}-{end_sec:.2f}s.{format}"
        filepath = output_dir / filename
        
        sf.write(filepath, segment, samplerate=sample_rate, subtype=subtype)
        saved_paths.append(filepath)

        logger.log("\nSaved audio chunk to: ", str(filepath), colors=["SUCCESS", "BRIGHT_SUCCESS"])
    
    return saved_paths


def merge_audio_chunks(
    chunk_files: Union[List[Path], Iterator[Path], List[str], Iterator[str]],
    output_path: Union[str, Path],
    overlap_duration: float = 2.0,
    sample_rate: int = 16000,
    expected_channels: int = 2,
    subtype: str = "PCM_16",
) -> Path:
    """
    Merge overlapping audio chunks into a single continuous WAV file.
    Correctly removes the overlapping region from all chunks except the first.

    This is the inverse of splitting with overlap — essential for reconstructing
    the original clean stream from chunked real-time recordings.

    Args:
        chunk_files: Ordered list or iterator of chunk file paths (as saved by save_audio_chunks)
        output_path: Destination path for the merged file
        overlap_duration: How much overlap was used when splitting (seconds)
        sample_rate: Sample rate of the chunks (must match)
        expected_channels: Number of channels (1 or 2)
        subtype: Output format subtype (e.g. "PCM_16", "PCM_24")

    Returns:
        Path to the saved merged file

    Example:
        >>> merged = merge_audio_chunks(saved_paths, "output/full_stream.wav", overlap_duration=2.0)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlap_samples = int(sample_rate * overlap_duration)
    consolidated_chunks: List[np.ndarray] = []

    chunk_paths = [Path(p) for p in chunk_files] if not isinstance(chunk_files, (list, tuple)) else list(chunk_files)

    if len(chunk_paths) == 0:
        raise ValueError("No chunk files provided to merge")

    logger.info(f"Merging {len(chunk_paths)} chunks → {output_path.name} (removing {overlap_duration}s overlap)")

    for idx, chunk_path in enumerate(chunk_paths):
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        audio, sr = sf.read(str(chunk_path))
        if sr != sample_rate:
            logger.warning(f"Resampling chunk {chunk_path.name} from {sr} → {sample_rate} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Ensure correct channels
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1) if expected_channels == 2 else audio[:, np.newaxis]
        elif audio.shape[1] != expected_channels:
            audio = np.tile(audio[:, :1], (1, expected_channels))

        audio = audio.astype(np.float32)

        if idx == 0:
            # First chunk: keep full
            consolidated_chunks.append(audio)
        else:
            # All other chunks: drop the trailing overlap
            if len(audio) > overlap_samples:
                consolidated_chunks.append(audio[:-overlap_samples])
            else:
                logger.warning(
                    f"Chunk {idx} ({chunk_path.name}) is shorter than overlap ({len(audio)} < {overlap_samples} samples). "
                    "Including full chunk to avoid gaps."
                )
                consolidated_chunks.append(audio)

    # Final concatenation
    merged_audio = np.concatenate(consolidated_chunks, axis=0)

    # Write output
    sf.write(
        str(output_path),
        merged_audio,
        samplerate=sample_rate,
        subtype=subtype,
        format="WAV",
    )

    duration = len(merged_audio) / sample_rate
    logger.info(f"Merged stream saved: {output_path} ({duration:.2f}s, {len(merged_audio):,} samples)")

    return output_path


def merge_in_memory_chunks(
    chunks_with_times: Iterator[Tuple[np.ndarray, float, float]],
    overlap_duration: float = 0.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Merge list of (audio_segment, start, end) with overlap removal (in-memory only)."""
    overlap_samples = int(sample_rate * overlap_duration)
    merged = []
    for i, (seg, start, end) in enumerate(chunks_with_times):
        if i == 0:
            merged.append(seg)
        else:
            merged.append(seg[:-overlap_samples] if len(seg) > overlap_samples else seg)
    return np.concatenate(merged, axis=0)

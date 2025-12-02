from pathlib import Path
import sys
import subprocess
import os
import glob
import re
from typing import Iterator, List, Sequence, Tuple, TypedDict, Union
import librosa
import sounddevice as sd
import numpy as np
from jet.logger import logger
import soundfile as sf  # <-- fast, supports many formats, free & popular
from numpy.typing import NDArray

# Supported audio extensions (common + comprehensive)
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm", ".mp4", ".mkv", ".avi"
}

AudioInput = Union[str, Path, Sequence[Union[str, Path]]]

def resolve_audio_paths(audio_inputs: AudioInput, recursive: bool = False) -> List[Path]:
    """Resolve single/list/dir → flat list of existing audio file Paths.

    Args:
        audio_inputs: Single path, list of paths, or directory.
        recursive: If True and a directory is given, walk recursively.
                   Defaults to False (only direct children).

    Returns:
        Flat list of Path objects pointing to valid audio files.
    """
    inputs = [audio_inputs] if isinstance(audio_inputs, (str, Path)) else audio_inputs
    resolved: List[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            audio_files = [
                p for p in iterator
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
            ]
            if not audio_files:
                logger.warning(f"No audio files found in directory: {path}")
            resolved.extend(audio_files)
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            resolved.append(path)
        elif path.exists():
            logger.warning(f"Skipping non-audio or unsupported file: {path}")
        else:
            logger.error(f"Path not found: {path}")

    if not resolved:
        raise ValueError("No valid audio files found from provided inputs.")

    return sorted(resolved, key=lambda p: p.resolve())

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


class AudioSegment(TypedDict):
    segment: np.ndarray
    start_time: float       # absolute start time in the original audio (seconds)
    end_time: float         # absolute end time in the original audio (seconds)
    segment_index: int      # 0-based index of this chunk
    is_first: bool          # True only for the very first segment
    is_last: bool           # True only for the final segment (may be shorter)
    overlaps_previous: bool # True if this segment overlaps with the previous chunk
    overlaps_next: bool     # True if this segment will overlap with the next chunk
                            # (always True except possibly the last one when overlap > 0)


def split_audio(
    audio: Union[np.ndarray, str, Path],
    segment_duration: float = 20.0,
    overlap_duration: float = 2.0,
    sample_rate: int = 16000,
) -> Iterator[AudioSegment]:
    """
    Yield richly annotated audio segments with precise overlap information.
    """
    if isinstance(audio, (str, Path)):
        audio, sr = librosa.load(str(audio), sr=sample_rate, mono=True)
    else:
        sr = sample_rate

    seg_samples = int(sr * segment_duration)
    ovl_samples = int(sr * overlap_duration)
    step_samples = seg_samples - ovl_samples

    if step_samples <= 0:
        raise ValueError("overlap_duration must be less than segment_duration")

    total_samples = len(audio)
    pos = 0
    segment_index = 0

    while pos < total_samples:
        end = min(pos + seg_samples, total_samples)
        segment = audio[pos:end]

        start_sec = pos / sr
        end_sec = end / sr

        # Determine overlap flags
        is_first = (segment_index == 0)
        is_last = (end == total_samples)
        overlaps_previous = (not is_first) and (ovl_samples > 0)
        overlaps_next = (not is_last) and (ovl_samples > 0)

        yield AudioSegment(
            segment=segment,
            start_time=start_sec,
            end_time=end_sec,
            segment_index=segment_index,
            is_first=is_first,
            is_last=is_last,
            overlaps_previous=overlaps_previous,
            overlaps_next=overlaps_next,
        )

        pos += step_samples
        segment_index += 1


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


AudioChunk = NDArray[np.floating]  # covers float32/float64 from librosa/sf

def _natural_sort_key(path: Path) -> list:
    """Natural sort: chunk_9.wav before chunk_10.wav"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if sr != target_sr:
        logger.debug(f"Resampling {path.name}: {sr} → {target_sr} Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def merge_audio_chunks(
    chunks: Union[
        str, Path,                              # single file or directory
        AudioChunk,                             # single in-memory array
        Sequence[Union[str, Path]],             # list of files
        Iterator[Union[str, Path]],             # iterator of files
        Sequence[AudioChunk],                   # list of in-memory arrays
    ],
    output_path: Union[str, Path] = "merged_output.wav",
    *,
    overlap_duration: float = 0.0,
    sample_rate: int = 16000,
    expected_channels: int = 2,
    subtype: str = "PCM_16",
    recursive: bool = True,
    audio_extensions: Tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"),
) -> Path:
    """
    Universal audio chunk merger.

    Pass:
      - A directory path → auto-find + natural sort all audio files recursively
      - A list of file paths → use exactly those (in order)
      - A single or list of np.ndarray → merge in-memory chunks

    Examples:
        merge_audio_chunks("recordings/session_2025/")                    # auto-discover
        merge_audio_chunks(["part1.wav", "part2.wav"])                   # explicit files
        merge_audio_chunks([array1, array2, array3])                     # in-memory
        merge_audio_chunks(np.random.rand(320000, 2))                    # single array
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap_samples = int(sample_rate * overlap_duration)

    # ------------------- Resolve input -------------------
    if isinstance(chunks, (str, Path)):
        path = Path(chunks).expanduser().resolve()
        if path.is_dir():
            # Directory mode
            pattern = "**/*.*" if recursive else "*.*"
            files = [
                f for ext in audio_extensions
                for f in path.glob(pattern)
                if f.suffix.lower() in ext
            ]
            if not files:
                raise FileNotFoundError(f"No audio files found in directory: {path}")
            files.sort(key=_natural_sort_key)
            arrays = [_load_audio(f, sample_rate) for f in files]
            source = f"directory '{path}' ({len(files)} files)"
        else:
            # Single file
            arrays = [_load_audio(path, sample_rate)]
            source = f"single file '{path}'"
    elif isinstance(chunks, np.ndarray):
        arrays = [np.asarray(chunks, dtype=np.float32)]
        source = "single in-memory array"
    elif isinstance(chunks, Sequence) and chunks and isinstance(chunks[0], np.ndarray):
        arrays = [np.asarray(arr, dtype=np.float32) for arr in chunks]
        source = f"{len(arrays)} in-memory arrays"
    else:
        # List/iterator of file paths
        paths = [Path(p).resolve() for p in chunks]
        if not all(p.exists() for p in paths):
            missing = [p for p in paths if not p.exists()]
            raise FileNotFoundError(f"Missing files: {missing[:5]}{'...' if len(missing)>5 else ''}")
        arrays = [_load_audio(p, sample_rate) for p in paths]
        source = f"{len(paths)} explicit file(s)"

    logger.info(f"Merging {len(arrays)} chunks from {source} → {output_path.name}")

    # ------------------- Process chunks -------------------
    processed: List[np.ndarray] = []
    for i, audio in enumerate(arrays):
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        # Normalize channels
        if audio.shape[1] == 1 and expected_channels == 2:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > expected_channels:
            audio = audio[:, :expected_channels]

        if i == 0:
            processed.append(audio)
        else:
            keep = len(audio) - overlap_samples
            if keep > 0:
                processed.append(audio[:keep])
            else:
                logger.warning(f"Chunk {i} too short for overlap ({len(audio)} < {overlap_samples} samples), keeping full")
                processed.append(audio)

    merged = np.concatenate(processed, axis=0)

    # ------------------- Write -------------------
    sf.write(str(output_path), merged, samplerate=sample_rate, subtype=subtype)
    duration = len(merged) / sample_rate
    logger.log(f"Merged ({duration:.2f}s, {merged.shape}): ", str(output_path), colors=["SUCCESS", "BRIGHT_SUCCESS"])

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

# jet/audio/utils/combine.py

from pathlib import Path
from typing import List, Literal, Optional

from pydub import AudioSegment

ChannelStrategy = Literal["mono", "stereo", "match-first"]


def _resolve_target_channels(
    first_segment: AudioSegment,
    strategy: ChannelStrategy,
) -> int:
    """
    Resolve target channel count based on strategy.
    """
    if strategy == "mono":
        return 1
    if strategy == "stereo":
        return 2
    if strategy == "match-first":
        return first_segment.channels

    raise ValueError(f"Unsupported channel strategy: {strategy}")


def _normalize_segment(
    segment: AudioSegment,
    sample_rate: int,
    sample_width: int,
    channels: int,
) -> AudioSegment:
    """
    Normalize audio segment to target format.
    """
    return (
        segment.set_frame_rate(sample_rate)
        .set_sample_width(sample_width)
        .set_channels(channels)
    )


def combine_audio_files(
    input_paths: List[Path],
    output_path: Path,
    sample_rate: int = 16000,
    dtype: Literal["int16", "int32"] = "int16",
    format: Optional[str] = None,
    bitrate: Optional[str] = None,
    channel_strategy: ChannelStrategy = "mono",
) -> None:
    """
    Concatenate multiple audio files with normalization.

    Args:
        input_paths: List of input audio files
        output_path: Output file
        sample_rate: Target sample rate
        dtype: int16 or int32
        format: Output format
        bitrate: Optional bitrate
        channel_strategy:
            - "mono": force 1 channel
            - "stereo": force 2 channels
            - "match-first": match first input file
    """
    if not input_paths:
        raise ValueError("No input audio files provided")

    for p in input_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")

    if dtype not in ("int16", "int32"):
        raise ValueError(f"Unsupported dtype '{dtype}'")

    sample_width = 2 if dtype == "int16" else 4

    first_seg = AudioSegment.from_file(str(input_paths[0]))
    target_channels = _resolve_target_channels(first_seg, channel_strategy)

    combined = _normalize_segment(first_seg, sample_rate, sample_width, target_channels)

    for path in input_paths[1:]:
        seg = AudioSegment.from_file(str(path))
        seg = _normalize_segment(seg, sample_rate, sample_width, target_channels)
        combined += seg

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = output_path.suffix.lstrip(".") if output_path.suffix else "wav"

    export_params = {"format": format}
    if bitrate:
        export_params["bitrate"] = bitrate

    combined.export(str(output_path), **export_params)

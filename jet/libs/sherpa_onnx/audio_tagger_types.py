from pathlib import Path
from typing import List, Optional, TypedDict, Union


class TaggingResult(TypedDict):
    """Typed dictionary for audio tagging results."""

    index: int
    name: str
    class_index: int
    prob: float


class ChunkTaggingResult(TypedDict):
    """Per-chunk tagging result with timing metadata."""

    chunk_index: int
    start_time: float
    end_time: float
    duration: float
    predictions: List[TaggingResult]
    processing_time: float


class AudioChunksTaggingSummary(TypedDict):
    """Complete summary of chunked audio tagging."""

    audio_path: str
    total_duration: float
    sample_rate: int
    chunk_duration: float
    overlap_duration: float
    total_chunks: int
    chunks: List[ChunkTaggingResult]
    overall_top_predictions: List[TaggingResult]
    total_processing_time: float
    real_time_factor: float


class AudioTaggerConfig(TypedDict, total=False):
    """Typed dictionary for AudioTagger configuration."""

    model_path: Optional[Union[str, Path]]
    labels_path: Optional[Union[str, Path]]
    top_k: int
    num_threads: int
    provider: str
    debug: bool
    speech_prob_threshold: float
    speech_top_n: int
    # Chunking defaults (from jet.audio.helpers.config)
    chunk_duration: float  # seconds
    chunk_overlap: float  # seconds
    min_chunk_duration: float  # seconds


class AudioTaggingSummary(TypedDict):
    """Typed dictionary for audio tagging summary."""

    audio_path: str
    duration_seconds: float
    sample_rate: int
    num_results: int
    top_predictions: List[TaggingResult]
    speech_detected: bool
    max_speech_probability: float
    processing_time_seconds: float
    real_time_factor: float

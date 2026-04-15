import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
from jet.audio.audio_types import AudioInput
from jet.audio.speech.ten_vad.speech_types import SpeechSegment
from jet.audio.utils import load_audio
from rich.console import Console
from ten_vad import TenVad

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio in range [-1, 1] to int16."""
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return audio_int16


def extract_speech_timestamps(
    audio: AudioInput,
    with_scores: bool = False,
    include_non_speech: bool = False,
    hop_size: int = 256,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech segments from audio using TEN VAD.

    Args:
        audio: Input audio (file path, bytes, numpy array, or torch tensor).
        with_scores: If True, return a tuple (segments, frame_probs).
        include_non_speech: If True, include non-speech segments in output.
        hop_size: Number of samples per VAD frame (default 256).
        threshold: VAD probability threshold for speech (default 0.5).
        min_speech_duration_ms: Minimum duration of a speech segment in ms.
        min_silence_duration_ms: Minimum duration of silence to split segments.

    Returns:
        List of SpeechSegment dicts, or tuple (segments, frame_probs).
    """
    audio_np, sr = load_audio(audio, sr=16000, mono=True)
    if sr != 16000:
        raise ValueError(f"TEN VAD requires 16000 Hz, got {sr}")

    # Convert to int16 as required by TenVad
    audio_int16 = float32_to_int16(audio_np)

    # Initialize VAD
    vad = TenVad(hop_size=hop_size, threshold=threshold)

    # Process audio frame by frame
    num_samples = len(audio_int16)
    num_frames = (num_samples + hop_size - 1) // hop_size  # ceil division
    probs = []
    for i in range(num_frames):
        start = i * hop_size
        end = min(start + hop_size, num_samples)
        frame = audio_int16[start:end]
        if len(frame) < hop_size:
            # Pad with zeros if last frame is shorter
            frame = np.pad(frame, (0, hop_size - len(frame)), mode="constant")
        prob, flag = vad.process(frame)
        probs.append(prob)

    # Convert probabilities to binary voice activity
    vad_flags = [1 if p >= threshold else 0 for p in probs]

    # Convert frame-level decisions to segments
    segments = []
    frame_duration_ms = (hop_size / sr) * 1000

    # Merge consecutive speech frames into segments, applying min durations
    i = 0
    segment_counter = 0
    while i < len(vad_flags):
        if vad_flags[i] == 1:
            # Start of a potential speech segment
            start_frame = i
            while i < len(vad_flags) and vad_flags[i] == 1:
                i += 1
            end_frame = i - 1
            # Check duration
            duration_frames = end_frame - start_frame + 1
            duration_ms = duration_frames * frame_duration_ms
            if duration_ms >= min_speech_duration_ms:
                segment = {
                    "num": segment_counter,
                    "start": start_frame * frame_duration_ms / 1000.0,
                    "end": (end_frame + 1) * frame_duration_ms / 1000.0,
                    "prob": float(np.mean(probs[start_frame : end_frame + 1])),
                    "duration": duration_ms / 1000.0,
                    "frames_length": duration_frames,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "type": "speech",
                    "segment_probs": probs[start_frame : end_frame + 1],
                }
                segments.append(segment)
                segment_counter += 1
        else:
            # Non-speech frame
            if include_non_speech:
                start_frame = i
                while i < len(vad_flags) and vad_flags[i] == 0:
                    i += 1
                end_frame = i - 1
                duration_frames = end_frame - start_frame + 1
                duration_ms = duration_frames * frame_duration_ms
                # Optionally filter short non-speech segments? Not required.
                segment = {
                    "num": segment_counter,
                    "start": start_frame * frame_duration_ms / 1000.0,
                    "end": (end_frame + 1) * frame_duration_ms / 1000.0,
                    "prob": float(np.mean(probs[start_frame : end_frame + 1])),
                    "duration": duration_ms / 1000.0,
                    "frames_length": duration_frames,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "type": "non-speech",
                    "segment_probs": probs[start_frame : end_frame + 1],
                }
                segments.append(segment)
                segment_counter += 1
            else:
                i += 1

    if with_scores:
        return segments, probs
    return segments


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="VAD probability threshold"
    )
    parser.add_argument(
        "-s", "--hop-size", type=int, default=256, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech", action="store_true", help="Include non-speech segments"
    )

    args = parser.parse_args()

    segments, scores = extract_speech_timestamps(
        audio=args.input,
        include_non_speech=args.include_non_speech,
        hop_size=args.hop_size,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
        with_scores=True,
    )

    speech_segments_json = Path(args.output) / "speech_segments.json"
    speech_probs_json = Path(args.output) / "speech_probs.json"
    segments_dir = Path(args.output) / "segments"

    segments_dir.mkdir(parents=True, exist_ok=True)

    with open(speech_segments_json, "w") as f:
        json.dump(
            {
                "count": len(segments),
                "segments": segments,
            },
            f,
            indent=2,
        )
    segments_url = f"file://{speech_segments_json.resolve()}"
    console.print(
        f"[green]{len(segments)} Segments saved to [link={segments_url}]{speech_segments_json}[/link][/green]"
    )

    with open(speech_probs_json, "w") as f:
        json.dump(scores, f, indent=2)

    probs_url = f"file://{speech_probs_json.resolve()}"
    console.print(
        f"[green]{len(scores)} probabilities saved to [link={probs_url}]{speech_probs_json}[/link][/green]"
    )

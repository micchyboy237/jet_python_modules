from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import List, Literal

from jet.audio.audio_types import AudioInput
from jet.audio.helpers.config import HOP_SIZE, SAMPLE_RATE
from jet.audio.speech.firered.speech_types import SpeechWave
from jet.audio.utils.loader import load_audio

WaveState = Literal["below", "above"]


def get_speech_waves(
    audio: AudioInput,
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = SAMPLE_RATE,
) -> List[SpeechWave]:
    """
    Identify complete speech waves (rise → sustained high → fall) from FireRedVAD probabilities.

    This function now accepts any AudioInput type and internally uses load_audio()
    for consistent preprocessing (though the audio itself is not processed further here
    unless you need to derive probabilities).
    """
    # Load audio for consistency (ensures correct sample rate and format)
    # We don't use the waveform here directly, but loading it ensures validation
    # and normalization of the input.
    _, loaded_sr = load_audio(audio, sr=sampling_rate, mono=True)

    # Use the full probability list
    all_waves = check_speech_waves(
        speech_probs=speech_probs,
        threshold=threshold,
        sampling_rate=loaded_sr,  # Use the confirmed sample rate
    )

    # Filter only valid (complete) waves
    valid_waves: List[SpeechWave] = []
    for wave in all_waves:
        if wave.get("is_valid", False):
            valid_waves.append(wave)

    return valid_waves


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = SAMPLE_RATE,
) -> List[SpeechWave]:
    """
    Analyze speech probabilities from FireRedVAD and return complete wave metadata.
    Updated for 10ms hop length (HOP_SIZE samples per frame).
    """
    if not speech_probs:
        return []

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    # Handle case where probabilities start already above threshold
    if speech_probs and speech_probs[0] >= threshold:
        current_wave = SpeechWave(
            has_risen=False,
            has_multi_passed=False,
            has_fallen=False,
            is_valid=False,
            start_sec=0.0,
            end_sec=0.0,
            details={
                "frame_start": 0,
                "frame_end": 0,
                "frame_len": 0,
                "duration_sec": 0.0,
                "min_prob": speech_probs[0],
                "max_prob": speech_probs[0],
                "avg_prob": speech_probs[0],
                "std_prob": 0.0,
            },
        )
        state = "above"

    for i, prob in enumerate(speech_probs):
        # Frame time in seconds using FireRedVAD hop size (10ms)
        frame_time_sec = i * HOP_SIZE / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i
                current_wave = SpeechWave(
                    has_risen=True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=frame_time_sec,
                    end_sec=frame_time_sec,
                    details={
                        "frame_start": i,
                        "frame_end": i,
                        "frame_len": 0,
                        "duration_sec": 0.0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "avg_prob": prob,
                        "std_prob": 0.0,
                    },
                )
                state = "above"

        else:  # state == "above"
            if prob >= threshold:
                if current_wave is not None:
                    current_wave["has_multi_passed"] = True
            else:
                if current_wave is not None:
                    current_wave["has_fallen"] = True
                    current_wave["is_valid"] = (
                        current_wave["has_risen"] and current_wave["has_multi_passed"]
                    )
                    current_wave["end_sec"] = frame_time_sec

                    # Finalize details for complete wave
                    frame_start = rise_frame_idx if rise_frame_idx is not None else 0
                    frame_end = i
                    wave_probs = speech_probs[frame_start:frame_end]
                    frame_len = frame_end - frame_start

                    current_wave["details"] = {
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "frame_len": frame_len,
                        "duration_sec": current_wave["end_sec"]
                        - current_wave["start_sec"],
                        "min_prob": min(wave_probs),
                        "max_prob": max(wave_probs),
                        "avg_prob": statistics.mean(wave_probs),
                        "std_prob": statistics.stdev(wave_probs)
                        if frame_len > 1
                        else 0.0,
                    }
                    waves.append(current_wave)

                current_wave = None
                rise_frame_idx = None
                state = "below"

    # Handle unfinished wave at the end of the sequence
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False  # incomplete waves are never valid
        current_wave["end_sec"] = len(speech_probs) * HOP_SIZE / sampling_rate

        if rise_frame_idx is not None:
            frame_start = rise_frame_idx
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start

            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "duration_sec": current_wave["end_sec"] - current_wave["start_sec"],
                "min_prob": min(wave_probs),
                "max_prob": max(wave_probs),
                "avg_prob": statistics.mean(wave_probs),
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
            }
        waves.append(current_wave)

    return waves


if __name__ == "__main__":
    import argparse

    from jet.audio.speech.firered.speech_timestamps_extractor import (
        extract_speech_timestamps,
    )
    from jet.file.utils import save_file

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Input audio file path (default: {DEFAULT_AUDIO})",
    )
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
        "-s", "--hop-size", type=int, default=160, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        "-d",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-g",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments",
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

    speech_waves = get_speech_waves(args.input, scores)

    save_file(segments, OUTPUT_DIR / "segments.json")
    save_file(scores, OUTPUT_DIR / "speech_probs.json")
    save_file(speech_waves, OUTPUT_DIR / "speech_waves.json")

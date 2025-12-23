from __future__ import annotations

from typing import List, Tuple, Literal
import statistics
from jet.audio.speech.silero.speech_types import SpeechSegment, SpeechWave

WaveState = Literal["below", "above"]

SpeechWaveTuple = Tuple[float, float]

def get_speech_waves(
    seg: SpeechSegment,
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = 16000,
) -> List[SpeechWaveTuple]:
    """
    Identify complete speech waves (rise → sustained high → fall) within a Silero VAD segment.

    Returns a list of (wave_start_seconds, wave_end_seconds) tuples representing full waves
    that cross the threshold upward, stay above for some time, and then cross downward.
    """
    if not speech_probs:
        return []

    # Handle both sample-based and second-based timestamps
    if isinstance(seg["start"], float) and isinstance(seg["end"], float):
        start_sample = int(round(seg["start"] * sampling_rate))
        end_sample = int(round(seg["end"] * sampling_rate))
    else:
        start_sample = int(seg["start"])
        end_sample = int(seg["end"])

    window_size_samples = 512 if sampling_rate == 16000 else 256

    start_idx = max(0, start_sample // window_size_samples)
    end_idx = min(len(speech_probs), (end_sample + window_size_samples - 1) // window_size_samples)

    if start_idx >= end_idx:
        return []

    segment_probs = speech_probs[start_idx:end_idx]

    # Use check_speech_waves on the segment slice
    all_waves = check_speech_waves(
        speech_probs=segment_probs,
        threshold=threshold,
        sampling_rate=sampling_rate,
    )

    # Filter only complete valid waves and convert to simple tuples
    valid_waves: List[SpeechWaveTuple] = []
    for wave in all_waves:
        if wave["is_valid"]:
            # Adjust start_sec to account for the segment offset
            adjusted_start = wave["start_sec"] + (start_idx * window_size_samples / sampling_rate)
            adjusted_end = wave["end_sec"] + (start_idx * window_size_samples / sampling_rate)
            valid_waves.append((adjusted_start, adjusted_end))

    return valid_waves


def check_speech_waves(
    speech_probs: List[float],
    threshold: float = 0.5,
    sampling_rate: int = 16000,
) -> List[SpeechWave]:
    """
    Analyze speech probabilities and return complete wave metadata with timestamps.
    Each returned SpeechWave includes timing information (start_sec, end_sec) in seconds.
    """
    if not speech_probs:
        return []

    window_size_samples = 512 if sampling_rate == 16000 else 256
    samples_per_frame = window_size_samples

    waves: List[SpeechWave] = []
    current_wave: SpeechWave | None = None
    state: WaveState = "below"
    rise_frame_idx: int | None = None

    # Handle case where probabilities start already above threshold
    if speech_probs and speech_probs[0] >= threshold:
        frame_time_sec = 0.0
        current_wave = SpeechWave(
            has_risen=False,
            has_multi_passed=False,
            has_fallen=False,
            is_valid=False,
            start_sec=frame_time_sec,
            end_sec=frame_time_sec,
            details={
                "frame_start": 0,
                "frame_end": 0,  # temporary, will update if extended
                "frame_len": 0,
                "min_prob": speech_probs[0],
                "max_prob": speech_probs[0],
                "mean_prob": speech_probs[0],
                "std_prob": 0.0,
            },
        )
        state = "above"

    for i, prob in enumerate(speech_probs):
        frame_time_sec = i * samples_per_frame / sampling_rate

        if state == "below":
            if prob >= threshold:
                rise_frame_idx = i
                current_wave = SpeechWave(
                    has_risen=True,
                    has_multi_passed=False,
                    has_fallen=False,
                    is_valid=False,
                    start_sec=frame_time_sec,
                    end_sec=frame_time_sec,  # will update on fall
                    details={
                        "frame_start": i,
                        "frame_end": i,  # temporary
                        "frame_len": 0,
                        "min_prob": prob,
                        "max_prob": prob,
                        "mean_prob": prob,
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
                    current_wave["is_valid"] = current_wave["has_risen"] and current_wave["has_multi_passed"]
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
                        "min_prob": min(wave_probs),
                        "max_prob": max(wave_probs),
                        "mean_prob": statistics.mean(wave_probs),
                        "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
                    }
                    waves.append(current_wave)
                current_wave = None
                rise_frame_idx = None
                state = "below"

    # Handle unfinished wave at end of sequence
    if current_wave is not None:
        current_wave["has_fallen"] = False
        current_wave["is_valid"] = False
        # Explicitly: incomplete waves are never valid
        current_wave["end_sec"] = len(speech_probs) * samples_per_frame / sampling_rate
        # Finalize details for trailing incomplete wave
        if rise_frame_idx is not None:
            frame_start = rise_frame_idx if rise_frame_idx is not None else 0
            frame_end = len(speech_probs)
            wave_probs = speech_probs[frame_start:frame_end]
            frame_len = frame_end - frame_start
            current_wave["details"] = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_len": frame_len,
                "min_prob": min(wave_probs),
                "max_prob": max(wave_probs),
                "mean_prob": statistics.mean(wave_probs),
                "std_prob": statistics.stdev(wave_probs) if frame_len > 1 else 0.0,
            }
        waves.append(current_wave)

    return waves

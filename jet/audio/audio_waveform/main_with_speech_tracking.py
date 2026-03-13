"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from fireredvad.core.constants import SAMPLE_RATE
from jet.audio.audio_waveform.app_with_speech_tracking import (
    AudioWaveformWithSpeechProbApp,
)
from jet.audio.audio_waveform.speech_tracker import (
    SpeechSegment,
    StreamingSpeechTracker,
)
from jet.audio.audio_waveform.vad.firered import FireRedVADWrapper
from jet.file.utils import save_file
from jet.transformers.object import make_serializable

OUTPUT_DIR = Path(__file__).parent / "generated" / "speech_tracker"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def save_speech_segment(
    segment_audio: np.ndarray,
    segment: SpeechSegment,
    probs: list[float],
    sample_rate: int = SAMPLE_RATE,
) -> None:
    now_str = segment.created_at.strftime("%Y%m%d_%H%M%S")
    segment_dir = OUTPUT_DIR / "segments" / f"segment_{now_str}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    # Write audio
    wav_path = segment_dir / "sound.wav"
    sf.write(wav_path, segment_audio, sample_rate, subtype="PCM_16")

    # Write probabilities
    probs_path = segment_dir / "speech_probs.json"
    with open(probs_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "probs": [round(p, 4) for p in probs],
                "frame_shift_sec": 0.01,  # ← hardcoded; consider passing if it becomes configurable
            },
            f,
            indent=2,
        )

    # Write summary
    summary = {
        "start_sec": round(segment.start_sec, 3),
        "end_sec": round(segment.end_sec, 3),
        "duration_sec": round(segment.duration_sec, 3),
        "frame_start": segment.start_frame,
        "frame_end": segment.end_frame,
        "frames_total": segment.total_frames,
        "created_at": now_str,
        "forced_split": segment.forced_split,
        "prob_info": segment.prob_info,
    }
    summary_path = segment_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ SAVED segment ({segment.duration_sec:.2f}s) → {wav_path}")


def handle_on_speech(
    segment_audio: np.ndarray,
    segment: SpeechSegment,
    probs: list[float],
) -> None:
    pass
    # You can extend this later (logging, UI update, sending to server, etc.)


def main():
    def on_speech_completed(
        audio: np.ndarray,
        segment: SpeechSegment,
        probs: list[float],
    ):
        print("\n-------")
        print(f"🛑 SPEECH ENDED (~{segment.end_sec:.2f}s)")
        save_speech_segment(audio, segment, probs)
        if segment.forced_split:
            print("  (forced split due to max duration)")
        handle_on_speech(audio, segment, probs)

        all_segments = speech_tracker.get_all_segments()
        print(
            f"SPEECH SEGMENTS (~{len(all_segments)}):\n{json.dumps(make_serializable(all_segments), indent=2)}"
        )

        old_segments = speech_tracker.old_segments
        save_file(all_segments, OUTPUT_DIR / "segments.json")
        save_file(old_segments, OUTPUT_DIR / "old_segments.json")

        print("-------\n")

    vad_fr = FireRedVADWrapper(
        # min_speech_duration_sec=0.5,
        # min_silence_duration_sec=0.9,
        # max_speech_duration_sec=5.0,
        # merge_small_segments=True,
    )
    speech_tracker = StreamingSpeechTracker(
        vad=vad_fr.vad,
        on_speech=on_speech_completed,  # we handle saving ourselves
    )

    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
        vad=vad_fr,
        speech_tracker=speech_tracker,
    )

    # Monkey-patch or override the tracker's on_speech behaviour
    # (alternative: pass our own callback wrapper from the beginning)
    def wrapped_update(*args, **kwargs):
        result = speech_tracker.update(*args, **kwargs)
        if result is not None:
            # We don't have direct access to probs & audio here → need small refactor
            # For clean solution see speech_tracker.py changes below
            pass
        return result

    app.start()


if __name__ == "__main__":
    main()

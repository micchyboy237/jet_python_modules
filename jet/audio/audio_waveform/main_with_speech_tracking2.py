#!/usr/bin/env python3
"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

import shutil
from pathlib import Path

from jet.audio.audio_waveform.app_with_speech_tracking2 import (
    AudioWaveformWithSpeechProbApp,
)

OUTPUT_DIR = Path(__file__).parent / "generated"
SAVED_SPEECH_SEGMENTS_DIR = OUTPUT_DIR / "saved_speech_segments"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
SAVED_SPEECH_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
        speech_save_dir=str(SAVED_SPEECH_SEGMENTS_DIR),
    )
    app.start()


if __name__ == "__main__":
    main()

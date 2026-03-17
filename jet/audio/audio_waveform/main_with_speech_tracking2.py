"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

import shutil
from pathlib import Path

from jet.audio.audio_waveform.app_with_speech_tracking2 import (
    AudioWaveformWithSpeechProbApp,
)
from jet.audio.audio_waveform.speech_handlers.speech_segment_saver import (
    SpeechSegmentSaver,
)

OUTPUT_DIR = Path(__file__).parent / "generated"
SAVED_SPEECH_SEGMENTS_DIR = OUTPUT_DIR / "saved_speech_segments"
shutil.rmtree(SAVED_SPEECH_SEGMENTS_DIR, ignore_errors=True)
SAVED_SPEECH_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
    )

    # Register the default file-saving handler
    saver = SpeechSegmentSaver(base_save_dir=SAVED_SPEECH_SEGMENTS_DIR)
    app.tracker.add_handler(saver)

    # You can add more handlers here later, e.g.:
    # app.tracker.add_handler(ConsoleLoggerHandler())
    # app.tracker.add_handler(UploadHandler(...))

    app.start()


if __name__ == "__main__":
    main()

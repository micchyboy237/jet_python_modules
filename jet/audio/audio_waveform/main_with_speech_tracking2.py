"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

import shutil
from pathlib import Path

from jet.audio.audio_waveform.app_with_speech_tracking2 import (
    AudioWaveformWithSpeechProbApp,
)
from jet.audio.audio_waveform.speech_handlers.live_srt_preview_handler import (
    LiveSrtPreviewHandler,
)
from jet.audio.audio_waveform.speech_handlers.speech_segment_saver import (
    SpeechSegmentSaver,
)
from jet.audio.audio_waveform.speech_handlers.websocket_subtitle_sender import (
    SubtitleEntry,
    WebsocketSubtitleSender,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / "speech_tracking"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

SAVED_SPEECH_SEGMENTS_DIR = OUTPUT_DIR / "saved_speech_segments"
SAVED_SPEECH_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SRT_PATH = OUTPUT_DIR / "live_subtitles.srt"


def main():
    min_silence_frame = 90
    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
        min_silence_frame=min_silence_frame,
    )

    saver = SpeechSegmentSaver(base_save_dir=SAVED_SPEECH_SEGMENTS_DIR)

    subtitle_entries = SubtitleEntry(output_path=GLOBAL_SRT_PATH)

    # WebSocket sender (binary + UUID matching)
    ws_handler = WebsocketSubtitleSender(
        accumulator=subtitle_entries,
        debug_save_audio=True,
    )

    # New: real-time preview handler
    preview_handler = LiveSrtPreviewHandler(accumulator=subtitle_entries)

    # Register handlers
    app.tracker.add_handler(saver)
    app.tracker.add_handler(ws_handler)
    app.tracker.add_handler(preview_handler)  # ← added

    def graceful_shutdown():
        print("\nShutting down gracefully...")
        ws_handler.close()
        preview_handler.close()

    import signal

    signal.signal(signal.SIGINT, lambda sig, frame: graceful_shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: graceful_shutdown())

    try:
        app.start()
    except Exception as e:
        print(f"Main loop exception: {e}")
        graceful_shutdown()


if __name__ == "__main__":
    main()

import sys
from typing import Literal

from jet.audio.speech.speechbrain.speech_types import SpeechSegment
from jet.overlays.speech_logger_overlay.overlay import LoggerOverlay
from jet.overlays.speech_logger_overlay.table_builder import build_speech_segments_table
from PyQt6.QtWidgets import QApplication

LogLevel = Literal["debug", "info", "warning", "error", "success"]


if __name__ == "__main__":
    app = QApplication(sys.argv)

    logger = LoggerOverlay.create_logger()

    segments: list[SpeechSegment] = [
        {
            "num": 1,
            "start": 0.0,
            "end": 820.5,
            "prob": 0.982,
            "frame_start": 0,
            "frame_end": 82,
            "type": "speech",
        },
        {
            "num": 2,
            "start": 820.5,
            "end": 1200.0,
            "prob": 0.112,
            "frame_start": 82,
            "frame_end": 120,
            "type": "non-speech",
        },
        {
            "num": 3,
            "start": 1200.0,
            "end": 2650.4,
            "prob": 0.945,
            "frame_start": 120,
            "frame_end": 265,
            "type": "speech",
        },
        {
            "num": 4,
            "start": 2650.4,
            "end": 3100.0,
            "prob": 0.201,
            "frame_start": 265,
            "frame_end": 310,
            "type": "non-speech",
        },
        {
            "num": 5,
            "start": 3100.0,
            "end": 4825.8,
            "prob": 0.991,
            "frame_start": 310,
            "frame_end": 482,
            "type": "speech",
        },
    ]

    table_html = build_speech_segments_table(segments)
    logger.html(table_html)

    sys.exit(QApplication.exec())

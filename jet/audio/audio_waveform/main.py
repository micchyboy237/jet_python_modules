#!/usr/bin/env python3
"""
Realtime audio waveform + speech probability visualizer (entry point)
"""

from jet.audio.audio_waveform.app import AudioWaveformWithSpeechProbApp


def main():
    app = AudioWaveformWithSpeechProbApp(
        samplerate=16000,
        block_size=512,
        display_points=200,
    )
    app.start()


if __name__ == "__main__":
    main()

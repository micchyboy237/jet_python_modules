"""
Main application logic — coordinates audio, VADs and UI updates
"""

from __future__ import annotations

import signal
import sys
import threading
from queue import Empty, Queue

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from jet.audio.audio_waveform.circular_buffer import CircularBuffer
from jet.audio.audio_waveform.plots import create_plots_layout
from jet.audio.audio_waveform.speech_tracker2 import SpeechSegmentTracker
from jet.audio.audio_waveform.vad.firered_with_speech_tracking import FireRedVADWrapper
from jet.audio.audio_waveform.vad.silero import SileroVAD
from jet.audio.audio_waveform.vad.speechbrain import SpeechBrainVADWrapper
from pyqtgraph.Qt import QtCore, QtWidgets


class AudioWaveformWithSpeechProbApp:
    def __init__(
        self,
        samplerate: int = 16000,
        block_size: int = 512,
        display_points: int = 200,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.5,
        pad_start_frame: int = 5,
        min_speech_frame: int = 30,
        soft_max_speech_frame: int = 450,
        hard_max_speech_frame: int = 800,
        min_silence_frame: int = 20,
        search_window: int = 200,
        valley_threshold: float = 0.65,
        chunk_max_frame: int = 30000,
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.display_points = display_points

        # Thresholds
        self.THRES_WAVE_MEDIUM = 0.01
        self.THRES_WAVE_HIGH = 0.15
        self.THRES_PROB_MEDIUM = 0.3
        self.THRES_PROB_HIGH = 0.7

        # VAD models
        self.vad = SileroVAD(samplerate=self.samplerate)
        self.vad_sb = SpeechBrainVADWrapper()

        self.tracker = SpeechSegmentTracker()
        self.vad_fr = FireRedVADWrapper(
            tracker=self.tracker,
            smooth_window_size=smooth_window_size,
            speech_threshold=speech_threshold,
            pad_start_frame=pad_start_frame,
            min_speech_frame=min_speech_frame,
            soft_max_speech_frame=soft_max_speech_frame,
            hard_max_speech_frame=hard_max_speech_frame,
            min_silence_frame=min_silence_frame,
            search_window=search_window,
            valley_threshold=valley_threshold,
            chunk_max_frame=chunk_max_frame,
        )

        # Thread-safe queue
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=50)

        # Data buffers
        self.wave_buffer = CircularBuffer(display_points)
        self.prob_buffer = CircularBuffer(display_points)
        self.prob_sb_buffer = CircularBuffer(display_points)
        self.prob_fr_buffer = CircularBuffer(display_points)

        self._init_buffers_with_zeros()

        # Qt setup
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)

        signal.signal(signal.SIGINT, lambda sig, frame: self.app.quit())

        pg.setConfigOptions(useOpenGL=True)

        # Create UI
        (
            self.win,
            (
                self.wave_curves,
                self.prob_curves,
                self.prob_sb_curves,
                self.prob_fr_curves,
            ),
        ) = create_plots_layout()

        # Position window bottom-right
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        margin = 0
        x = screen.width() - self.win.width() - margin
        y = screen.height() - self.win.height() - 70 - margin
        self.win.move(max(0, x), max(0, y))
        self.win.show()

        # Audio input
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # Background worker
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )
        self.worker_thread.start()

        # UI update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(30)

    def _init_buffers_with_zeros(self) -> None:
        for _ in range(self.display_points):
            self.wave_buffer.append(0.0)
            self.prob_buffer.append(0.0)
            self.prob_sb_buffer.append(0.0)
            self.prob_fr_buffer.append(0.0)

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)
        samples = indata[:, 0].astype(np.float32)
        try:
            self.audio_queue.put_nowait(samples)
        except:
            pass  # drop if full

    def _inference_worker(self) -> None:
        while self._running:
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            wave_value = np.max(np.abs(samples)) if samples.size > 0 else 0.0
            self.wave_buffer.append(wave_value)

            prob = self.vad.get_speech_prob(samples)
            self.prob_buffer.append(prob)

            prob_sb = self.vad_sb.get_speech_prob(samples)
            self.prob_sb_buffer.append(prob_sb)

            prob_fr = self.vad_fr.get_speech_prob(samples)
            self.prob_fr_buffer.append(prob_fr)

            if self.tracker:
                self.tracker.add_audio(samples)  # ← feed raw mic audio

    def _update_plots(self) -> None:
        self._update_one_plot(
            self.wave_buffer,
            self.wave_curves,
            self.THRES_WAVE_MEDIUM,
            self.THRES_WAVE_HIGH,
            is_waveform=True,
        )

        self._update_one_plot(
            self.prob_buffer,
            self.prob_curves,
            self.THRES_PROB_MEDIUM,
            self.THRES_PROB_HIGH,
        )

        self._update_one_plot(
            self.prob_sb_buffer,
            self.prob_sb_curves,
            self.THRES_PROB_MEDIUM,
            self.THRES_PROB_HIGH,
        )

        self._update_one_plot(
            self.prob_fr_buffer,
            self.prob_fr_curves,
            self.THRES_PROB_MEDIUM,
            self.THRES_PROB_HIGH,
        )

    def _update_one_plot(
        self,
        buffer: CircularBuffer,
        curves: tuple,
        thresh_medium: float,
        thresh_high: float,
        is_waveform: bool = False,
    ) -> None:
        data = buffer.to_array()
        if len(data) == 0:
            return

        x = np.arange(len(data), dtype=np.float32)

        low_mask = data < thresh_medium
        mid_mask = (data >= thresh_medium) & (data < thresh_high)
        high_mask = data >= thresh_high

        # Connect segments smoothly
        for mask in (low_mask, mid_mask, high_mask):
            mask[:-1] |= mask[1:]
            mask[1:] |= mask[:-1]

        low = np.where(low_mask, data, np.nan)
        mid = np.where(mid_mask, data, np.nan)
        high = np.where(high_mask, data, np.nan)

        low_curve, mid_curve, high_curve = curves
        low_curve.setData(x, low)
        mid_curve.setData(x, mid)
        high_curve.setData(x, high)

    def start(self) -> None:
        with self.stream:
            self.app.exec()
        self._running = False

"""
Realtime audio waveform + speech probability visualizer.

Three stacked plots:
1. Waveform
2. Speech probability (0–1) — Silero VAD
3. Speech probability (0–1) — SpeechBrain CRDNN VAD

Dependencies:
    pip install sounddevice numpy pyqtgraph torch
    pip install speechbrain          # for the third plot
"""

from __future__ import annotations

import signal
import sys
import threading
from collections import deque
from queue import Empty, Queue
from typing import Deque

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
import torch
from pyqtgraph.Qt import QtCore, QtWidgets

# -----------------------------------------------------------------------------
# Circular Buffer Base
# -----------------------------------------------------------------------------


class CircularBuffer:
    """Generic fixed-length circular buffer."""

    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        self.max_len = max_len
        self._buffer: Deque[float] = deque(maxlen=max_len)

    def append(self, values: np.ndarray | float) -> None:
        if isinstance(values, np.ndarray):
            for v in values:
                self._buffer.append(float(v))
        else:
            self._buffer.append(float(values))

    def to_array(self) -> np.ndarray:
        return np.array(self._buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)


# -----------------------------------------------------------------------------
# Silero VAD Wrapper
# -----------------------------------------------------------------------------


class SileroVAD:
    """Thin wrapper around Silero VAD model for streaming usage."""

    def __init__(self, samplerate: int = 16000, device: str | None = None) -> None:
        if samplerate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 Hz or 16000 Hz")

        self.samplerate = samplerate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading Silero VAD on {self.device}... ", end="", flush=True)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.to(self.device)
        self.model.eval()

        torch.set_num_threads(1)  # recommended for desktop/low-core-count

        print("done.")

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Compute speech probability for one audio block.
        chunk: float32, shape (n_samples,)
        """
        if chunk.ndim != 1:
            raise ValueError("Expected 1D audio chunk")

        tensor = torch.from_numpy(chunk).float().to(self.device).unsqueeze(0)
        prob = self.model(tensor, self.samplerate).item()
        return prob


# -----------------------------------------------------------------------------
# SpeechBrain VAD Wrapper
# -----------------------------------------------------------------------------


class SpeechBrainVADWrapper:
    """Wrapper for speechbrain vad-crdnn-libriparty with simple streaming-like API."""

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(
            f"Loading SpeechBrain VAD (vad-crdnn-libriparty) on {self.device}... ",
            end="",
            flush=True,
        )
        from speechbrain.inference.VAD import VAD

        self.vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
            run_opts={"device": str(self.device)},
        )
        self.vad.eval()
        print("done.")

        self.sample_rate = 16000  # required by model
        self.context_samples = int(
            0.5 * self.sample_rate
        )  # ~0.5 s context — good trade-off
        self.audio_ring: torch.Tensor = torch.zeros(
            self.context_samples, dtype=torch.float32, device=self.device
        )
        self.write_pos = 0

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0

        # Append new chunk to ring buffer (overwrite oldest)
        chunk_t = torch.from_numpy(chunk).float().to(self.device)
        chunk_len = len(chunk_t)
        space = self.context_samples - self.write_pos
        if chunk_len <= space:
            self.audio_ring[self.write_pos : self.write_pos + chunk_len] = chunk_t
            self.write_pos += chunk_len
        else:
            self.audio_ring[:chunk_len] = chunk_t[-self.context_samples :]
            self.write_pos = chunk_len

        # Run inference on current context
        prob_tensor = self.vad.get_speech_prob_chunk(self.audio_ring.unsqueeze(0))
        return float(prob_tensor[-1, -1].item())  # last frame of last chunk


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------


class AudioWaveformWithSpeechProbApp:
    """
    Displays:
        - Waveform (top)
        - Speech probability (middle, Silero VAD)
        - Speech probability (bottom, SpeechBrain CRDNN VAD)
    """

    def __init__(
        self,
        samplerate: int = 16000,
        block_size: int = 512,
        display_points: int = 200,  # ≈ 6.4 seconds @ 512 samples / 32 ms per block
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.display_points = display_points

        # Thresholds for coloring waveform and probability
        # Waveform ranges:
        #   < WAVE_MEDIUM  → gray
        #   WAVE_MEDIUM ≤ x < WAVE_HIGH  → cyan
        #   ≥ WAVE_HIGH              → green
        self.THRES_WAVE_MEDIUM = 0.01
        self.THRES_WAVE_HIGH = 0.15

        # Speech probability ranges
        self.THRES_PROB_MEDIUM = 0.3
        self.THRES_PROB_HIGH = 0.7

        self.vad = SileroVAD(samplerate=self.samplerate)
        self.vad_sb = SpeechBrainVADWrapper()  # speechbrain vad

        # Thread-safe audio queue (prevents blocking audio callback)
        self.audio_queue: Queue[np.ndarray] = Queue(maxsize=50)

        # Buffers
        self.wave_buffer = CircularBuffer(display_points)
        self.prob_buffer = CircularBuffer(display_points)
        self.prob_sb_buffer = CircularBuffer(display_points)

        # Initialize with zeros for smooth startup visual
        for _ in range(display_points):
            self.wave_buffer.append(0.0)
            self.prob_buffer.append(0.0)
            self.prob_sb_buffer.append(0.0)

        # Qt App
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)

        def _quit_on_sigint(sig, frame):
            self.app.quit()

        signal.signal(signal.SIGINT, _quit_on_sigint)

        # Enable OpenGL for smoother performance
        pg.setConfigOptions(useOpenGL=True)

        # Prepare flags first
        flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint

        # Layout window
        self.win = pg.GraphicsLayoutWidget(
            show=False,  # ← important: do NOT show yet
            size=(450, 300),
            title="Realtime Audio + Speech Probability",
        )
        self.win.setWindowFlags(flags)

        # -------------------------
        # Waveform plot (TOP)
        # -------------------------
        self.wave_plot = self.win.addPlot()
        self.wave_plot.setYRange(0, 1.1)
        self.wave_plot.setLabel("left", "Audio Amp")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.15)  # Add faint grid
        self.wave_curve_low = self.wave_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.2),
            name="low",
            connect="finite",
        )
        self.wave_curve_mid = self.wave_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=1.8),
            name="mid",
            connect="finite",
        )
        self.wave_curve_high = self.wave_plot.plot(
            pen=pg.mkPen(color=(100, 255, 120), width=2.2),
            name="high",
            connect="finite",
        )

        # Move to next row
        self.win.nextRow()

        # -------------------------
        # Speech Probability plot (MIDDLE, Silero)
        # -------------------------
        self.prob_plot = self.win.addPlot()
        self.prob_plot.setYRange(0, 1)
        self.prob_plot.setLabel("left", "Speech Prob")
        self.prob_plot.showGrid(x=True, y=True, alpha=0.15)  # Add faint grid
        self.prob_curve_low = self.prob_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.2),
            name="low",
            connect="finite",
        )
        self.prob_curve_mid = self.prob_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=1.8),
            name="mid",
            connect="finite",
        )
        self.prob_curve_high = self.prob_plot.plot(
            pen=pg.mkPen(color=(100, 255, 120), width=2.2),
            name="high",
            connect="finite",
        )
        # Optional: keep for debugging
        # self.prob_plot.addLegend()

        # Move to next row for SpeechBrain plot
        self.win.nextRow()

        # -------------------------
        # SpeechBrain VAD Probability plot (BOTTOM)
        # -------------------------
        self.prob_sb_plot = self.win.addPlot(title="SpeechBrain VAD Prob")
        self.prob_sb_plot.setYRange(0, 1)
        self.prob_sb_plot.setLabel("left", "SB Speech Prob")
        self.prob_sb_plot.showGrid(x=True, y=True, alpha=0.15)
        self.prob_sb_low = self.prob_sb_plot.plot(
            pen=pg.mkPen(color=(180, 150, 180), width=1.2)
        )
        self.prob_sb_mid = self.prob_sb_plot.plot(
            pen=pg.mkPen(color=(200, 100, 200), width=1.8)
        )
        self.prob_sb_high = self.prob_sb_plot.plot(
            pen=pg.mkPen(color=(220, 60, 220), width=2.2)
        )

        # Position at bottom-right corner with small margin
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        margin = 0
        x = screen.width() - self.win.width() - margin
        y = screen.height() - self.win.height() - 70 - margin
        self.win.move(max(0, x), max(0, y))  # prevent going off-screen left/top

        self.win.show()  # show explicitly after flags & position

        # Audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # Start background inference thread
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )
        self.worker_thread.start()

        # UI timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(30)

    # -------------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)
        samples = indata[:, 0].astype(np.float32)
        try:
            self.audio_queue.put_nowait(samples)
        except:
            pass  # drop if queue full (better than blocking audio thread)

    # -------------------------------------------------------------------------

    def _inference_worker(self) -> None:
        """
        Runs VAD inference off the audio thread.
        """
        while self._running:
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            # Waveform peak
            wave_value = np.max(np.abs(samples)) if samples.size > 0 else 0.0
            self.wave_buffer.append(wave_value)

            # Silero VAD
            prob = self.vad.get_speech_prob(samples)
            self.prob_buffer.append(prob)

            # SpeechBrain VAD
            prob_sb = self.vad_sb.get_speech_prob(samples)
            self.prob_sb_buffer.append(prob_sb)

    # -------------------------------------------------------------------------

    def _update_plots(self) -> None:
        wave_data = self.wave_buffer.to_array()
        if len(wave_data) > 0:
            x = np.arange(len(wave_data), dtype=np.float32)

            # Create masks
            low_mask = wave_data < self.THRES_WAVE_MEDIUM
            mid_mask = (wave_data >= self.THRES_WAVE_MEDIUM) & (
                wave_data < self.THRES_WAVE_HIGH
            )
            high_mask = wave_data >= self.THRES_WAVE_HIGH

            # Ensure continuity by overlapping boundary points
            for mask in (low_mask, mid_mask, high_mask):
                mask[:-1] |= mask[1:]
                mask[1:] |= mask[:-1]

            low = np.where(low_mask, wave_data, np.nan)
            mid = np.where(mid_mask, wave_data, np.nan)
            high = np.where(high_mask, wave_data, np.nan)

            self.wave_curve_low.setData(x, low)
            self.wave_curve_mid.setData(x, mid)
            self.wave_curve_high.setData(x, high)

        prob_data = self.prob_buffer.to_array()
        if len(prob_data) > 0:
            x_prob = np.arange(len(prob_data), dtype=np.float32)

            # Create masks
            low_mask = prob_data < self.THRES_PROB_MEDIUM
            mid_mask = (prob_data >= self.THRES_PROB_MEDIUM) & (
                prob_data < self.THRES_PROB_HIGH
            )
            high_mask = prob_data >= self.THRES_PROB_HIGH

            # Ensure continuity by overlapping boundary points
            for mask in (low_mask, mid_mask, high_mask):
                mask[:-1] |= mask[1:]
                mask[1:] |= mask[:-1]

            low = np.where(low_mask, prob_data, np.nan)
            mid = np.where(mid_mask, prob_data, np.nan)
            high = np.where(high_mask, prob_data, np.nan)

            self.prob_curve_low.setData(x_prob, low)
            self.prob_curve_mid.setData(x_prob, mid)
            self.prob_curve_high.setData(x_prob, high)

        # ── SpeechBrain prob plot ───────────────────────────────────────
        prob_sb_data = self.prob_sb_buffer.to_array()
        if len(prob_sb_data) > 0:
            x_prob = np.arange(len(prob_sb_data), dtype=np.float32)

            low_mask = prob_sb_data < self.THRES_PROB_MEDIUM
            mid_mask = (prob_sb_data >= self.THRES_PROB_MEDIUM) & (
                prob_sb_data < self.THRES_PROB_HIGH
            )
            high_mask = prob_sb_data >= self.THRES_PROB_HIGH

            for mask in (low_mask, mid_mask, high_mask):
                mask[:-1] |= mask[1:]
                mask[1:] |= mask[:-1]

            low = np.where(low_mask, prob_sb_data, np.nan)
            mid = np.where(mid_mask, prob_sb_data, np.nan)
            high = np.where(high_mask, prob_sb_data, np.nan)

            self.prob_sb_low.setData(x_prob, low)
            self.prob_sb_mid.setData(x_prob, mid)
            self.prob_sb_high.setData(x_prob, high)

    # -------------------------------------------------------------------------

    def start(self) -> None:
        with self.stream:
            self.app.exec()
        self._running = False


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app = AudioWaveformWithSpeechProbApp()
    app.start()

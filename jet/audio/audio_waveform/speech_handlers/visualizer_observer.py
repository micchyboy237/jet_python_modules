import sys
from typing import Callable, List

import numpy as np
from jet.audio.audio_waveform.circular_buffer import CircularBuffer
from jet.audio.audio_waveform.plots import create_plots_layout
from jet.audio.helpers.energy import smooth_signal
from pyqtgraph.Qt import QtCore, QtWidgets


class VisualizerObserver:
    def __init__(self, display_points: int = 200):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.display_points = display_points
        # Smoothing for the RMS waveform only (makes the plot much cleaner)
        self.rms_smooth_window = 7  # ← easy to tune (5–9 recommended)
        self.THRES_WAVE = (0.1, 0.5)  # Normalized RMS thresholds
        self.THRES_PROB = (0.3, 0.7)
        self.buffers = {
            "wave": CircularBuffer(display_points),
            "silero": CircularBuffer(display_points),
            "sb": CircularBuffer(display_points),
            "fr": CircularBuffer(display_points),
            "ten_vad": CircularBuffer(display_points),
            "hybrid": CircularBuffer(display_points),  # ← new
        }
        self._fill_initial_buffers()

        # Listeners notified whenever the VAD dropdown changes.
        # Each callable receives the internal key string, e.g. "fr", "silero".
        self._vad_changed_callbacks: List[Callable[[str], None]] = []

        (
            self.main_widget,
            self.c_wave,
            self.c_vad,
            self.c_hybrid,
            self.vad_selector,
            self.vad_plot,
            self.hybrid_plot,  # returned twice — assign once; see note below
        ) = create_plots_layout()

        # Set initial VAD type
        self.current_vad = "fr"  # FireRed
        self._update_vad_label(self.current_vad)

        # Connect dropdown signal
        self.vad_selector.currentTextChanged.connect(self.on_vad_selection_changed)

        self._prob_weight: float = 0.5
        self._rms_weight: float = 0.5

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    def add_vad_changed_callback(self, cb: Callable[[str], None]) -> None:
        """Register a function to be called whenever the VAD selector changes.

        The callback receives the internal VAD key (e.g. "fr", "silero",
        "sb", "ten_vad") so the caller can immediately act on the change
        without polling ``viz.current_vad``.
        """
        if cb not in self._vad_changed_callbacks:
            self._vad_changed_callbacks.append(cb)

    def on_vad_selection_changed(self, text: str):
        """Handle dropdown selection."""
        vad_map = {
            "FireRed": "fr",
            "Silero": "silero",
            "SpeechBrain": "sb",
            "TEN VAD": "ten_vad",
        }
        self.current_vad = vad_map.get(text, "fr")
        self._update_vad_label(self.current_vad)
        # Notify every registered listener about the new VAD key.
        for cb in self._vad_changed_callbacks:
            try:
                cb(self.current_vad)
            except Exception as e:
                print(f"[VisualizerObserver] vad_changed callback error: {e}")

    def _update_vad_label(self, vad_key: str):
        """Update the left axis label of the VAD plot."""
        label_map = {
            "fr": "FireRed Prob",
            "silero": "Silero Prob",
            "sb": "SpeechBrain Prob",
            "ten_vad": "TEN VAD Prob",
        }
        self.vad_plot.setLabel("left", label_map.get(vad_key, "VAD Prob"))

    def _fill_initial_buffers(self):
        for buf in self.buffers.values():
            for _ in range(self.display_points):
                buf.append(0.0)

    def __call__(self, samples: np.ndarray):
        pass

    def push_data(
        self,
        wave: float,
        silero: float,
        sb: float,
        fr: float,
        ten_vad: float,
        hybrid: float = 0.0,  # kept for API compatibility but ignored — computed at render time
    ):
        """Store new data points from all observers."""

        self.buffers["wave"].append(wave)
        self.buffers["silero"].append(silero)
        self.buffers["sb"].append(sb)
        self.buffers["fr"].append(fr)
        self.buffers["ten_vad"].append(ten_vad)
        # Do NOT store hybrid here. It is computed fresh in update_plots()
        # using whichever VAD the user has selected right now.

    def update_plots(self):
        # Apply smoothing only to the RMS waveform plot
        self._update_curve(
            self.buffers["wave"], self.c_wave, *self.THRES_WAVE, apply_smooth=True
        )

        # Pick the right raw-VAD buffer for the middle plot
        if self.current_vad == "silero":
            buffer = self.buffers["silero"]
        elif self.current_vad == "sb":
            buffer = self.buffers["sb"]
        elif self.current_vad == "fr":
            buffer = self.buffers["fr"]
        elif self.current_vad == "ten_vad":
            buffer = self.buffers["ten_vad"]
        else:
            buffer = self.buffers["fr"]

        self._update_curve(buffer, self.c_vad, *self.THRES_PROB)

        # Recompute hybrid on-the-fly from stored raw buffers so that
        # changing the dropdown instantly refreshes the whole hybrid curve.
        vad_data = buffer.to_array()
        wave_data = self.buffers["wave"].to_array()
        min_len = min(len(vad_data), len(wave_data))
        if min_len > 0:
            hybrid_data = (
                self._prob_weight * vad_data[-min_len:]
                + self._rms_weight * wave_data[-min_len:]
            )
            self._update_curve_data(hybrid_data, self.c_hybrid, *self.THRES_PROB)

    def _update_curve(self, buffer, curves, t_med, t_high, apply_smooth: bool = False):
        """Update one of the pyqtgraph curves with optional smoothing."""
        data = buffer.to_array()
        if len(data) == 0:
            return

        if apply_smooth and len(data) > 1:
            # Smooth only the waveform (RMS) curve — VAD probs stay as-is
            data = smooth_signal(data, window=self.rms_smooth_window)

        x = np.arange(len(data), dtype=np.float32)
        low_m = data < t_med
        mid_m = (data >= t_med) & (data < t_high)
        high_m = data >= t_high
        for m in (low_m, mid_m, high_m):
            m[:-1] |= m[1:]
            m[1:] |= m[:-1]
        low_c, mid_c, high_c = curves
        low_c.setData(x, np.where(low_m, data, np.nan))
        mid_c.setData(x, np.where(mid_m, data, np.nan))
        high_c.setData(x, np.where(high_m, data, np.nan))

    def _update_curve_data(
        self, data: np.ndarray, curves, t_med, t_high, apply_smooth: bool = False
    ):
        """Same as _update_curve but accepts a pre-built numpy array instead of a CircularBuffer."""
        if len(data) == 0:
            return
        if apply_smooth and len(data) > 1:
            from jet.audio.helpers.energy import smooth_signal

            data = smooth_signal(data, window=self.rms_smooth_window)
        x = np.arange(len(data), dtype=np.float32)
        low_m = data < t_med
        mid_m = (data >= t_med) & (data < t_high)
        high_m = data >= t_high
        for m in (low_m, mid_m, high_m):
            m[:-1] |= m[1:]
            m[1:] |= m[:-1]
        low_c, mid_c, high_c = curves
        low_c.setData(x, np.where(low_m, data, np.nan))
        mid_c.setData(x, np.where(mid_m, data, np.nan))
        high_c.setData(x, np.where(high_m, data, np.nan))

    def close(self):
        self.timer.stop()
        self.main_widget.close()
        self.app.quit()

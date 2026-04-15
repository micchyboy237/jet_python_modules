import sys

import numpy as np
from jet.audio.audio_waveform.circular_buffer import CircularBuffer
from jet.audio.audio_waveform.plots import create_plots_layout
from pyqtgraph.Qt import QtCore, QtWidgets


class VisualizerObserver:
    def __init__(self, display_points: int = 200):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.display_points = display_points
        self.THRES_WAVE = (0.01, 0.15)
        self.THRES_PROB = (0.3, 0.7)
        self.buffers = {
            "wave": CircularBuffer(display_points),
            "silero": CircularBuffer(display_points),
            "sb": CircularBuffer(display_points),
            "fr": CircularBuffer(display_points),
            "ten_vad": CircularBuffer(display_points),
        }
        self._fill_initial_buffers()

        # Create UI and get references
        (
            self.main_widget,
            self.c_wave,
            self.c_vad,
            self.vad_selector,
            self.vad_plot,  # <-- Now we have the plot item directly
        ) = create_plots_layout()

        # Set initial VAD type
        self.current_vad = "fr"  # FireRed
        self._update_vad_label(self.current_vad)

        # Connect dropdown signal
        self.vad_selector.currentTextChanged.connect(self.on_vad_selection_changed)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

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
        self, wave: float, silero: float, sb: float, fr: float, ten_vad: float
    ):
        """Store new data points from all observers."""
        self.buffers["wave"].append(wave)
        self.buffers["silero"].append(silero)
        self.buffers["sb"].append(sb)
        self.buffers["fr"].append(fr)
        self.buffers["ten_vad"].append(ten_vad)

    def update_plots(self):
        self._update_curve(self.buffers["wave"], self.c_wave, *self.THRES_WAVE)

        # Update only the selected VAD plot
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

    def _update_curve(self, buffer, curves, t_med, t_high):
        data = buffer.to_array()
        if len(data) == 0:
            return
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

import argparse
import signal
import sys

import numpy as np
import pyloudnorm as pyln
import pyqtgraph as pg
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

OVERLAY_WIDTH = 520
OVERLAY_HEIGHT = 300
DEFAULT_MARGIN_RIGHT = 20
DEFAULT_MARGIN_BOTTOM = 50
DEFAULT_MARGIN_TOP = 20
DEFAULT_SAMPLERATE = 44100
DEFAULT_BLOCKSIZE = 512
DEFAULT_WINDOW_SIZE_SECONDS = 5.0
DEFAULT_AMPLITUDE_SCALE = 2.0
MIN_AMPLITUDE_SCALE = 0.1
MAX_AMPLITUDE_SCALE = 10.0
AMPLITUDE_SCALE_STEP = 0.1
MAX_GAIN_DB = 30.0
SOFT_CLIP_THRESHOLD = 0.95

TARGET_LUFS_PRESETS = {
    "Streaming (-14 LUFS)": -14.0,
    "Broadcast (-23 LUFS)": -23.0,
    "YouTube (-13 LUFS)": -13.0,
    "Spotify (-14 LUFS)": -14.0,
    "Cinema (-27 LUFS)": -27.0,
}


class LiveAudioWaveform(QWidget):
    """Live audio waveform visualization with LUFS-based loudness normalization.

    Features:
    - Real-time audio waveform display
    - ITU-R BS.1770 loudness measurement (LUFS)
    - Manual and automatic loudness normalization (default: on)
    - Soft clipping to prevent visual distortion
    - Peak level warnings
    - Loudness range display
    - Multiple target LUFS presets
    """

    audio_data_ready = pyqtSignal(np.ndarray)

    def __init__(
        self,
        device=None,
        channels: int = 1,
        samplerate: int = DEFAULT_SAMPLERATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
        amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
        enable_normalization: bool = True,
        verbose: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.verbose = verbose
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.buffer_size = int(window_size_seconds * samplerate)
        self.audio_buffer = np.zeros((self.buffer_size, self.channels))
        self.write_index = 0
        self.amplitude_scale = amplitude_scale
        self.auto_scale = False
        self.current_rms = 0.0
        self.current_lufs = float("-inf")
        self.target_lufs = -14.0
        self.loudness_normalization_enabled = enable_normalization
        self.soft_clip_enabled = True
        self.gain_linear = 1.0
        self.gain_db = 0.0
        self.update_counter = 0
        self.peak_warning_active = False
        self.last_peak_warning = 0
        self.clip_count = 0
        self.loudness_range = 0.0

        # Initialize pyloudnorm meter
        self.loudness_meter = pyln.Meter(samplerate)

        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(
                geom.right() - self.width() - DEFAULT_MARGIN_RIGHT,
                geom.bottom() - self.height() - DEFAULT_MARGIN_BOTTOM,
            )
        self._setup_ui()
        self.audio_data_ready.connect(self.update_plot)
        self.drag_position = None

        # Set initial state of auto-normalize checkbox
        if enable_normalization:
            self.auto_normalize_cb.setChecked(True)

        self._log(
            f"[INIT] LiveAudioWaveform initialized with "
            f"amplitude_scale={amplitude_scale:.1f}x, "
            f"target LUFS={self.target_lufs:.0f}, "
            f"normalization={'ON' if enable_normalization else 'OFF'}"
        )

    def _log(self, message):
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(message)

    def _setup_ui(self):
        """Set up the user interface components."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self._create_top_bar()
        self._create_controls_bar()
        self._create_loudness_bar()
        self._create_clipping_bar()
        self._create_plot_widget()
        self._log("[UI] Setup complete")

    def _create_top_bar(self):
        """Create the title bar with close button and peak warning indicator."""
        self.top_bar = QWidget()
        self.top_bar.setFixedHeight(28)
        self.top_bar.setStyleSheet(
            "background-color: #2b2b2b; border-bottom: 1px solid #444;"
        )
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(8, 0, 8, 0)
        self.title_label = QLabel("🎙️ Live Audio Waveform + LUFS Norm")
        self.title_label.setStyleSheet(
            "color: #e0e0e0; font-size: 13px; font-weight: bold;"
        )
        top_layout.addWidget(self.title_label)

        # Peak warning indicator
        self.peak_warning_label = QLabel("")
        self.peak_warning_label.setStyleSheet(
            "color: #ff3333; font-size: 11px; font-weight: bold;"
        )
        top_layout.addWidget(self.peak_warning_label)

        top_layout.addStretch()
        self.scale_label = QLabel(f"Scale: {self.amplitude_scale:.1f}x")
        self.scale_label.setStyleSheet(
            "color: #00ffff; font-size: 11px; font-weight: bold;"
        )
        top_layout.addWidget(self.scale_label)
        top_layout.addSpacing(10)
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #ff5555; color: white; border: none; "
            "font-weight: bold; font-size: 16px; border-radius: 12px; } "
            "QPushButton:hover { background-color: #ff3333; }"
        )
        self.close_btn.clicked.connect(self.close)
        top_layout.addWidget(self.close_btn)
        self.main_layout.addWidget(self.top_bar)

    def _create_controls_bar(self):
        """Create controls for adjusting waveform height."""
        self.controls_bar = QWidget()
        self.controls_bar.setFixedHeight(40)
        self.controls_bar.setStyleSheet(
            "background-color: #333333; border-bottom: 1px solid #444;"
        )
        controls_layout = QHBoxLayout(self.controls_bar)
        controls_layout.setContentsMargins(8, 4, 8, 4)
        controls_layout.setSpacing(8)
        amp_label = QLabel("Height:")
        amp_label.setStyleSheet("color: #cccccc; font-size: 11px; font-weight: bold;")
        controls_layout.addWidget(amp_label)
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(
            int(MIN_AMPLITUDE_SCALE * 100), int(MAX_AMPLITUDE_SCALE * 100)
        )
        self.amp_slider.setValue(int(self.amplitude_scale * 100))
        self.amp_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; } "
            "QSlider::handle:horizontal { background: #00ffff; width: 16px; margin: -5px 0; "
            "border-radius: 8px; } "
            "QSlider::sub-page:horizontal { background: #00cccc; border-radius: 3px; }"
        )
        self.amp_slider.valueChanged.connect(self._on_amplitude_slider_changed)
        controls_layout.addWidget(self.amp_slider, 1)
        self.amp_spinbox = QSpinBox()
        self.amp_spinbox.setRange(
            int(MIN_AMPLITUDE_SCALE * 100), int(MAX_AMPLITUDE_SCALE * 100)
        )
        self.amp_spinbox.setValue(int(self.amplitude_scale * 100))
        self.amp_spinbox.setSingleStep(10)
        self.amp_spinbox.setSuffix("%")
        self.amp_spinbox.setFixedWidth(70)
        self.amp_spinbox.setStyleSheet(
            "QSpinBox { background: #444; color: #00ffff; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px; font-weight: bold; }"
        )
        self.amp_spinbox.valueChanged.connect(self._on_amplitude_spinbox_changed)
        controls_layout.addWidget(self.amp_spinbox)
        self.auto_scale_cb = QCheckBox("Auto")
        self.auto_scale_cb.setStyleSheet(
            "QCheckBox { color: #cccccc; font-size: 11px; font-weight: bold; spacing: 4px; } "
            "QCheckBox::indicator { width: 16px; height: 16px; } "
            "QCheckBox::indicator:unchecked { background: #555; border: 2px solid #666; "
            "border-radius: 3px; } "
            "QCheckBox::indicator:checked { background: #00ffff; border: 2px solid #00cccc; "
            "border-radius: 3px; }"
        )
        self.auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        controls_layout.addWidget(self.auto_scale_cb)
        self.rms_label = QLabel("Level: -∞ dB")
        self.rms_label.setStyleSheet("color: #999999; font-size: 10px;")
        controls_layout.addWidget(self.rms_label)
        controls_layout.addStretch()
        self.main_layout.addWidget(self.controls_bar)
        self._log("[UI] Controls bar created")

    def _create_loudness_bar(self):
        """Create bar for loudness normalization controls."""
        self.loudness_bar = QWidget()
        self.loudness_bar.setFixedHeight(36)
        self.loudness_bar.setStyleSheet(
            "background-color: #2d2d2d; border-bottom: 1px solid #444;"
        )
        loudness_layout = QHBoxLayout(self.loudness_bar)
        loudness_layout.setContentsMargins(8, 4, 8, 4)
        loudness_layout.setSpacing(6)

        # Normalize button
        self.normalize_btn = QPushButton("🔊 Normalize")
        self.normalize_btn.setFixedHeight(26)
        self.normalize_btn.setStyleSheet(
            "QPushButton { background-color: #00aa44; color: white; border: none; "
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: bold; } "
            "QPushButton:hover { background-color: #00cc55; } "
            "QPushButton:pressed { background-color: #008833; } "
            "QPushButton:disabled { background-color: #555555; color: #999999; }"
        )
        self.normalize_btn.clicked.connect(self._on_normalize_clicked)
        self.normalize_btn.setToolTip("Apply LUFS normalization gain to audio")
        loudness_layout.addWidget(self.normalize_btn)

        # Reset button
        self.reset_gain_btn = QPushButton("↺ Reset")
        self.reset_gain_btn.setFixedHeight(26)
        self.reset_gain_btn.setStyleSheet(
            "QPushButton { background-color: #666666; color: white; border: none; "
            "border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: bold; } "
            "QPushButton:hover { background-color: #888888; } "
            "QPushButton:pressed { background-color: #444444; }"
        )
        self.reset_gain_btn.clicked.connect(self._on_reset_gain_clicked)
        self.reset_gain_btn.setToolTip("Reset gain to 0 dB")
        loudness_layout.addWidget(self.reset_gain_btn)

        # Target LUFS combo box
        target_label = QLabel("Target:")
        target_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        loudness_layout.addWidget(target_label)

        self.target_combo = QComboBox()
        self.target_combo.addItems(list(TARGET_LUFS_PRESETS.keys()))
        self.target_combo.setCurrentText("Streaming (-14 LUFS)")
        self.target_combo.setStyleSheet(
            "QComboBox { background: #444; color: #00ffff; border: 1px solid #555; "
            "border-radius: 3px; padding: 2px 5px; font-size: 10px; font-weight: bold; } "
            "QComboBox::drop-down { border: none; } "
            "QComboBox::down-arrow { image: none; border-left: 4px solid transparent; "
            "border-right: 4px solid transparent; border-top: 6px solid #00ffff; "
            "margin-right: 5px; } "
            "QComboBox QAbstractItemView { background: #444; color: #00ffff; "
            "selection-background-color: #00aaaa; }"
        )
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        loudness_layout.addWidget(self.target_combo)

        # Auto-normalize checkbox
        self.auto_normalize_cb = QCheckBox("Auto")
        self.auto_normalize_cb.setStyleSheet(
            "QCheckBox { color: #cccccc; font-size: 10px; font-weight: bold; spacing: 3px; } "
            "QCheckBox::indicator { width: 14px; height: 14px; } "
            "QCheckBox::indicator:unchecked { background: #555; border: 2px solid #666; "
            "border-radius: 3px; } "
            "QCheckBox::indicator:checked { background: #00aa44; border: 2px solid #008833; "
            "border-radius: 3px; }"
        )
        self.auto_normalize_cb.toggled.connect(self._on_auto_normalize_toggled)
        self.auto_normalize_cb.setToolTip(
            "Automatically adjust gain when audio is soft"
        )
        loudness_layout.addWidget(self.auto_normalize_cb)

        # Loudness display
        self.lufs_label = QLabel("LUFS: -∞")
        self.lufs_label.setStyleSheet("color: #999999; font-size: 10px;")
        loudness_layout.addWidget(self.lufs_label)

        self.gain_label = QLabel("Gain: +0.0 dB")
        self.gain_label.setStyleSheet("color: #999999; font-size: 10px;")
        loudness_layout.addWidget(self.gain_label)

        # Loudness range indicator
        self.range_label = QLabel("")
        self.range_label.setStyleSheet("color: #777777; font-size: 9px;")
        loudness_layout.addWidget(self.range_label)

        loudness_layout.addStretch()
        self.main_layout.addWidget(self.loudness_bar)
        self._log("[UI] Loudness bar created")

    def _create_clipping_bar(self):
        """Create bar for soft clipping and peak warning controls."""
        self.clipping_bar = QWidget()
        self.clipping_bar.setFixedHeight(30)
        self.clipping_bar.setStyleSheet(
            "background-color: #282828; border-bottom: 1px solid #444;"
        )
        clipping_layout = QHBoxLayout(self.clipping_bar)
        clipping_layout.setContentsMargins(8, 2, 8, 2)
        clipping_layout.setSpacing(6)

        # Soft clipping checkbox
        self.soft_clip_cb = QCheckBox("Soft Clip")
        self.soft_clip_cb.setChecked(self.soft_clip_enabled)
        self.soft_clip_cb.setStyleSheet(
            "QCheckBox { color: #cccccc; font-size: 10px; font-weight: bold; spacing: 3px; } "
            "QCheckBox::indicator { width: 14px; height: 14px; } "
            "QCheckBox::indicator:unchecked { background: #555; border: 2px solid #666; "
            "border-radius: 3px; } "
            "QCheckBox::indicator:checked { background: #ff9900; border: 2px solid #cc7700; "
            "border-radius: 3px; }"
        )
        self.soft_clip_cb.toggled.connect(self._on_soft_clip_toggled)
        self.soft_clip_cb.setToolTip("Apply soft clipping to prevent visual distortion")
        clipping_layout.addWidget(self.soft_clip_cb)

        # Clip count display
        self.clip_count_label = QLabel("")
        self.clip_count_label.setStyleSheet("color: #777777; font-size: 9px;")
        clipping_layout.addWidget(self.clip_count_label)

        # Threshold adjustment
        threshold_label = QLabel("Thresh:")
        threshold_label.setStyleSheet("color: #888888; font-size: 9px;")
        clipping_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 99)  # 0.50 to 0.99
        self.threshold_slider.setValue(int(SOFT_CLIP_THRESHOLD * 100))
        self.threshold_slider.setFixedWidth(80)
        self.threshold_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; } "
            "QSlider::handle:horizontal { background: #ff9900; width: 12px; margin: -4px 0; "
            "border-radius: 6px; } "
            "QSlider::sub-page:horizontal { background: #cc7700; border-radius: 2px; }"
        )
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        clipping_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel(f"{SOFT_CLIP_THRESHOLD:.2f}")
        self.threshold_label.setStyleSheet(
            "color: #ff9900; font-size: 9px; font-weight: bold;"
        )
        clipping_layout.addWidget(self.threshold_label)

        clipping_layout.addStretch()
        self.main_layout.addWidget(self.clipping_bar)
        self._log("[UI] Clipping bar created")

    def _create_plot_widget(self):
        """Create the waveform plot widget with fixed Y-axis range."""
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getPlotItem().vb.setMouseEnabled(x=True, y=False)
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)
        self._update_y_range()
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.curve = self.plot_widget.plot(pen=pg.mkPen("#00ffff", width=2.5))

        # Clipping indicator line
        self.clip_line_top = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen("#ff3333", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.clip_line_bottom = pg.InfiniteLine(
            pos=-1.0,
            angle=0,
            pen=pg.mkPen("#ff3333", width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.clip_line_top)
        self.plot_widget.addItem(self.clip_line_bottom)

        self._add_reference_lines()
        self.main_layout.addWidget(self.plot_widget, 1)
        self._log("[UI] Plot widget created with fixed Y-axis range")

    def _add_reference_lines(self):
        """Add horizontal reference lines for visual guidance."""
        center_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(center_line)
        self.top_line = pg.InfiniteLine(
            pos=self.amplitude_scale,
            angle=0,
            pen=pg.mkPen("#ff5555", width=1, style=Qt.PenStyle.DotLine),
        )
        self.bottom_line = pg.InfiniteLine(
            pos=-self.amplitude_scale,
            angle=0,
            pen=pg.mkPen("#ff5555", width=1, style=Qt.PenStyle.DotLine),
        )
        self.plot_widget.addItem(self.top_line)
        self.plot_widget.addItem(self.bottom_line)
        unity_top = pg.InfiniteLine(
            pos=1.0,
            angle=0,
            pen=pg.mkPen("#00ff00", width=1, style=Qt.PenStyle.DashLine),
        )
        unity_bottom = pg.InfiniteLine(
            pos=-1.0,
            angle=0,
            pen=pg.mkPen("#00ff00", width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(unity_top)
        self.plot_widget.addItem(unity_bottom)
        self._log("[UI] Reference lines added")

    def _update_reference_lines(self):
        """Update reference line positions based on current scale."""
        if hasattr(self, "top_line"):
            self.top_line.setPos(self.amplitude_scale)
        if hasattr(self, "bottom_line"):
            self.bottom_line.setPos(-self.amplitude_scale)
        if hasattr(self, "clip_line_top"):
            self.clip_line_top.setPos(SOFT_CLIP_THRESHOLD * self.amplitude_scale)
        if hasattr(self, "clip_line_bottom"):
            self.clip_line_bottom.setPos(-SOFT_CLIP_THRESHOLD * self.amplitude_scale)

    def _update_y_range(self):
        """Update the Y-axis range based on current amplitude scale."""
        y_max = self.amplitude_scale * 1.1
        self.plot_widget.setYRange(-y_max, y_max)
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)
        self._update_reference_lines()
        if hasattr(self, "scale_label"):
            self.scale_label.setText(f"Scale: {self.amplitude_scale:.1f}x")
        self._log(
            f"[RANGE] Y-axis range set to [-{y_max:.1f}, {y_max:.1f}] with scale {self.amplitude_scale:.1f}x"
        )

    def _on_amplitude_slider_changed(self, value):
        """Handle amplitude slider value change."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:
            self.amplitude_scale = scale
            self.amp_spinbox.blockSignals(True)
            self.amp_spinbox.setValue(int(scale * 100))
            self.amp_spinbox.blockSignals(False)
            self._update_y_range()
            self._log(
                f"[CONTROL] Amplitude scale changed to {scale:.2f}x ({value}%) via slider"
            )

    def _on_amplitude_spinbox_changed(self, value):
        """Handle amplitude spinbox value change."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:
            self.amplitude_scale = scale
            self.amp_slider.blockSignals(True)
            self.amp_slider.setValue(int(scale * 100))
            self.amp_slider.blockSignals(False)
            self._update_y_range()
            self._log(
                f"[CONTROL] Amplitude scale changed to {scale:.2f}x ({value}%) via spinbox"
            )

    def _on_auto_scale_toggled(self, checked):
        """Handle auto-scale toggle."""
        self.auto_scale = checked
        if checked:
            self.amp_slider.setEnabled(False)
            self.amp_spinbox.setEnabled(False)
            self._log("[AUTO] Auto-scale enabled")
        else:
            self.amp_slider.setEnabled(True)
            self.amp_spinbox.setEnabled(True)
            self.amplitude_scale = self.amp_slider.value() / 100.0
            self._update_y_range()
            self._log("[AUTO] Auto-scale disabled")

    def _on_target_changed(self, text):
        """Handle target LUFS preset change."""
        self.target_lufs = TARGET_LUFS_PRESETS[text]
        self._log(f"[LOUDNESS] Target changed to {text} = {self.target_lufs:.0f} LUFS")
        # Re-trigger normalization if auto is on
        if self.loudness_normalization_enabled:
            self._on_normalize_clicked()

    def _on_auto_normalize_toggled(self, checked):
        """Handle auto-normalize toggle."""
        self.loudness_normalization_enabled = checked
        if checked:
            self._log("[LOUDNESS] Auto-normalization enabled")
            self._update_gain_display()
        else:
            self._log("[LOUDNESS] Auto-normalization disabled")

    def _on_soft_clip_toggled(self, checked):
        """Handle soft clip toggle."""
        self.soft_clip_enabled = checked
        self._log(f"[CLIP] Soft clipping {'enabled' if checked else 'disabled'}")

    def _on_threshold_changed(self, value):
        """Handle soft clip threshold change."""
        global SOFT_CLIP_THRESHOLD
        SOFT_CLIP_THRESHOLD = value / 100.0
        self.threshold_label.setText(f"{SOFT_CLIP_THRESHOLD:.2f}")
        self._update_reference_lines()
        self._log(f"[CLIP] Threshold changed to {SOFT_CLIP_THRESHOLD:.2f}")

    def _soft_clip(self, data, threshold=SOFT_CLIP_THRESHOLD):
        """Apply soft clipping to prevent harsh visual distortion.

        Uses tanh-based soft clipping which smoothly saturates near the threshold.

        Args:
            data: numpy array of audio samples
            threshold: clipping threshold (0.0 to 1.0)

        Returns:
            numpy array with soft clipping applied
        """
        if not self.soft_clip_enabled:
            return data

        # Scale by threshold so tanh starts saturating near threshold
        scaled = data / threshold
        return np.tanh(scaled) * threshold

    def _check_peaks(self, data):
        """Check for peaks that would clip and update warning display.

        Args:
            data: numpy array of audio samples (after gain, before display scaling)
        """
        max_peak = np.max(np.abs(data))

        if max_peak > 0.95:
            self.clip_count += 1
            if not self.peak_warning_active:
                self.peak_warning_active = True
                self.peak_warning_label.setText(f"⚠ PEAK: {max_peak:.2f}")
                self.peak_warning_label.setStyleSheet(
                    "color: #ff3333; font-size: 11px; font-weight: bold;"
                )
                self._log(f"[WARNING] Digital clipping detected! Peak: {max_peak:.3f}")
            else:
                self.peak_warning_label.setText(f"⚠ PEAK: {max_peak:.2f}")

            self.last_peak_warning = self.update_counter
            self.clip_count_label.setText(f"Clips: {self.clip_count}")
            self.clip_count_label.setStyleSheet(
                "color: #ff3333; font-size: 9px; font-weight: bold;"
            )
        elif (
            self.peak_warning_active
            and (self.update_counter - self.last_peak_warning) > 10
        ):
            # Clear warning after 10 updates without clipping
            self.peak_warning_active = False
            self.peak_warning_label.setText("")
            self.clip_count_label.setStyleSheet("color: #777777; font-size: 9px;")

    def _calculate_rms(self, data):
        """Calculate RMS value of audio data."""
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(data))))

    def _calculate_db(self, rms):
        """Convert RMS to decibels."""
        if rms <= 0.000001:
            return float("-inf")
        return 20 * np.log10(rms)

    def _measure_loudness(self, audio_data):
        """Measure loudness of audio data using pyloudnorm.
        Measures on the RAW audio (before gain is applied).

        Args:
            audio_data: numpy array of audio samples

        Returns:
            float: loudness in LUFS
        """
        if len(audio_data) < 512:
            return float("-inf")

        try:
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            loudness = self.loudness_meter.integrated_loudness(audio_data)
            return float(loudness)
        except Exception as e:
            self._log(f"[LOUDNESS] Error measuring loudness: {e}")
            return float("-inf")

    def _calculate_normalization_gain(self, current_lufs, target_lufs):
        """Calculate gain needed to normalize audio.

        Args:
            current_lufs: Current loudness in LUFS
            target_lufs: Target loudness in LUFS

        Returns:
            tuple: (gain_db, gain_linear)
        """
        if current_lufs == float("-inf"):
            return 0.0, 1.0

        gain_db = target_lufs - current_lufs

        # Clamp gain to reasonable limits
        gain_db = np.clip(gain_db, -20.0, MAX_GAIN_DB)

        gain_linear = 10 ** (gain_db / 20.0)

        self._log(
            f"[LOUDNESS] Normalization: {current_lufs:.1f} LUFS → {target_lufs:.0f} LUFS, "
            f"gain={gain_db:.1f} dB ({gain_linear:.2f}x)"
        )

        return gain_db, gain_linear

    def _update_gain_display(self):
        """Update the gain display label."""
        if abs(self.gain_db) < 0.01:
            self.gain_label.setText("Gain: +0.0 dB")
            self.gain_label.setStyleSheet("color: #999999; font-size: 10px;")
        elif self.gain_db > 0:
            self.gain_label.setText(f"Gain: +{self.gain_db:.1f} dB")
            if self.gain_db > 20:
                color = "#ff9900"  # Warning: high gain
            elif self.gain_db > 12:
                color = "#ffdd00"
            else:
                color = "#00ff00"
            self.gain_label.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold;"
            )
        else:
            self.gain_label.setText(f"Gain: {self.gain_db:.1f} dB")
            self.gain_label.setStyleSheet(
                "color: #ff6666; font-size: 10px; font-weight: bold;"
            )

    def _on_normalize_clicked(self):
        """Handle manual normalize button click.
        Applies LUFS-based gain to the actual audio buffer."""
        self._log("[LOUDNESS] Manual normalization triggered")

        # Get current buffer data (raw, ungained)
        display_data = np.roll(self.audio_buffer[:, 0], -self.write_index)

        # Measure current loudness on raw audio
        current_lufs = self._measure_loudness(display_data)

        if current_lufs == float("-inf"):
            self._log("[LOUDNESS] Cannot normalize - audio too quiet or silent")
            return

        # Calculate required gain
        gain_db, gain_linear = self._calculate_normalization_gain(
            current_lufs, self.target_lufs
        )

        if abs(gain_db) < 0.5:
            self._log(
                "[LOUDNESS] Audio already near target level. No adjustment needed."
            )
            self.gain_db = 0.0
            self.gain_linear = 1.0
            self._update_gain_display()
            return

        # Apply gain to the audio buffer
        self.gain_db = gain_db
        self.gain_linear = gain_linear
        self._update_gain_display()

        self._log(
            f"[LOUDNESS] Applied {gain_db:+.1f} dB gain to audio (linear: {gain_linear:.2f}x)"
        )

        # Update range indicator
        self.loudness_range = self._calculate_loudness_range(display_data)
        self.range_label.setText(f"Range: {self.loudness_range:.1f} LU")

    def _on_reset_gain_clicked(self):
        """Reset audio gain to 0 dB (unity gain)."""
        self._log("[LOUDNESS] Gain reset to 0 dB")
        self.gain_db = 0.0
        self.gain_linear = 1.0
        self.clip_count = 0
        self.peak_warning_active = False
        self.peak_warning_label.setText("")
        self.clip_count_label.setText("")
        self._update_gain_display()
        self.range_label.setText("")

    def _calculate_loudness_range(self, audio_data):
        """Calculate short-term loudness range."""
        if len(audio_data) < self.samplerate * 0.4:  # Need at least 400ms
            return 0.0

        try:
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            # Calculate short-term loudness in 400ms chunks
            chunk_size = int(self.samplerate * 0.4)
            num_chunks = len(audio_data) // chunk_size

            if num_chunks < 2:
                return 0.0

            loudness_values = []
            for i in range(num_chunks):
                chunk = audio_data[i * chunk_size : (i + 1) * chunk_size]
                try:
                    lufs = self.loudness_meter.integrated_loudness(chunk)
                    if lufs > -70:  # Ignore silence
                        loudness_values.append(lufs)
                except:
                    pass

            if len(loudness_values) < 2:
                return 0.0

            # Return the range (max - min)
            return max(loudness_values) - min(loudness_values)
        except Exception as e:
            self._log(f"[LOUDNESS] Error calculating range: {e}")
            return 0.0

    def _update_rms_display(self, rms):
        """Update the RMS level display with color coding."""
        db_level = self._calculate_db(rms)
        if db_level == float("-inf"):
            self.rms_label.setText("Level: -∞ dB")
            self.rms_label.setStyleSheet("color: #666666; font-size: 10px;")
        else:
            # Show both the raw and gained RMS
            gained_rms = rms * self.gain_linear
            gained_db = self._calculate_db(gained_rms)

            if gained_db > -3:
                color = "#ff3333"
            elif gained_db > -6:
                color = "#ff9900"
            elif gained_db > -12:
                color = "#ffdd00"
            elif gained_db > -20:
                color = "#00ff00"
            elif gained_db > -40:
                color = "#00ccff"
            else:
                color = "#6666ff"

            if abs(self.gain_db) > 0.1:
                self.rms_label.setText(
                    f"Level: {gained_db:.1f} dB (raw: {db_level:.1f})"
                )
            else:
                self.rms_label.setText(f"Level: {db_level:.1f} dB")
            self.rms_label.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold;"
            )

    def _update_lufs_display(self, lufs):
        """Update the LUFS display with color coding."""
        if lufs == float("-inf"):
            self.lufs_label.setText("LUFS: -∞")
            self.lufs_label.setStyleSheet("color: #666666; font-size: 10px;")
        else:
            # Show effective LUFS after gain
            effective_lufs = lufs + self.gain_db

            self.lufs_label.setText(f"LUFS: {effective_lufs:.1f}")

            # Color code based on distance from target
            distance = effective_lufs - self.target_lufs
            if distance >= -1:
                color = "#00ff00"  # Green: at or above target
            elif distance >= -3:
                color = "#ffdd00"  # Yellow: slightly below
            elif distance >= -6:
                color = "#ff9900"  # Orange: moderately below
            else:
                color = "#ff3333"  # Red: significantly below target

            self.lufs_label.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: bold;"
            )

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.close_btn.geometry().contains(event.pos()):
                return
            if self.controls_bar.geometry().contains(event.pos()):
                return
            if hasattr(self, "loudness_bar") and self.loudness_bar.geometry().contains(
                event.pos()
            ):
                return
            if hasattr(self, "clipping_bar") and self.clipping_bar.geometry().contains(
                event.pos()
            ):
                return
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for window dragging."""
        if (
            event.buttons() == Qt.MouseButton.LeftButton
            and self.drag_position is not None
        ):
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self.drag_position = None
        event.accept()

    def audio_callback(self, indata, frames, time, status):
        """Runs in a background thread. Must be fast and non-blocking."""
        if status:
            self._log(f"[AUDIO] Status: {status}")
        self.audio_data_ready.emit(indata.copy())

    def update_plot(self, new_data):
        """Runs in the main GUI thread. Updates the circular buffer and plot."""
        self.update_counter += 1
        new_frames = new_data.shape[0]

        # Circular buffer update
        if self.write_index + new_frames <= self.buffer_size:
            self.audio_buffer[self.write_index : self.write_index + new_frames] = (
                new_data
            )
            self.write_index += new_frames
        else:
            remaining = self.buffer_size - self.write_index
            self.audio_buffer[self.write_index :] = new_data[:remaining]
            self.audio_buffer[: new_frames - remaining] = new_data[remaining:]
            self.write_index = (self.write_index + new_frames) % self.buffer_size

        # Get display data from RAW buffer
        raw_display_data = np.roll(self.audio_buffer[:, 0], -self.write_index)

        # Apply audio gain (this is the actual "loudening")
        gained_data = raw_display_data * self.gain_linear

        # Check for peaks before soft clipping
        self._check_peaks(gained_data)

        # Apply soft clipping for visualization
        gained_data = self._soft_clip(gained_data)

        # Calculate RMS on raw and gained data
        raw_rms = self._calculate_rms(raw_display_data[-new_frames:])
        self._update_rms_display(raw_rms)

        # Measure loudness on RAW data periodically
        if self.update_counter % 20 == 0:
            self.current_lufs = self._measure_loudness(raw_display_data)
            self._update_lufs_display(self.current_lufs)

            # Auto-normalization
            if self.loudness_normalization_enabled and self.current_lufs != float(
                "-inf"
            ):
                effective_lufs = self.current_lufs + self.gain_db

                if effective_lufs < self.target_lufs - 0.5:
                    # Need more gain
                    gain_db_needed = self.target_lufs - self.current_lufs
                    gain_db_needed = np.clip(gain_db_needed, -20.0, MAX_GAIN_DB)

                    # Smooth transition
                    self.gain_db = self.gain_db * 0.8 + gain_db_needed * 0.2
                    self.gain_linear = 10 ** (self.gain_db / 20.0)
                    self._update_gain_display()

                    self._log(
                        f"[AUTO-NORM] Auto-adjusting: raw LUFS={self.current_lufs:.1f}, "
                        f"effective={effective_lufs:.1f}, target={self.target_lufs:.0f}, "
                        f"gain={self.gain_db:.1f} dB"
                    )
                elif effective_lufs > self.target_lufs + 2.0:
                    # Too loud, reduce gain
                    self.gain_db = self.gain_db * 0.9
                    self.gain_linear = 10 ** (self.gain_db / 20.0)
                    self._update_gain_display()

        # Auto-scale (existing functionality)
        if self.auto_scale and raw_rms > 0.0001:
            target_scale = (self.amplitude_scale * 0.7) / max(
                raw_rms * self.gain_linear, 0.0001
            )
            self.amplitude_scale = self.amplitude_scale * 0.95 + target_scale * 0.05
            self.amplitude_scale = float(
                np.clip(self.amplitude_scale, MIN_AMPLITUDE_SCALE, MAX_AMPLITUDE_SCALE)
            )
            self._update_y_range()

        # Update plot with gained data and display scale
        scaled_data = gained_data * self.amplitude_scale
        self.curve.setData(scaled_data)

        if self.update_counter % 50 == 0:
            self._log(
                f"[PLOT] Update #{self.update_counter}: "
                f"scale={self.amplitude_scale:.2f}x, "
                f"gain={self.gain_db:.1f}dB, "
                f"raw RMS={raw_rms:.4f}, "
                f"eff RMS={raw_rms * self.gain_linear:.4f}, "
                f"raw LUFS={self.current_lufs:.1f}, "
                f"eff LUFS={self.current_lufs + self.gain_db:.1f}, "
                f"peak={np.max(np.abs(scaled_data)):.3f}, "
                f"clips={self.clip_count}"
            )

    def start(self):
        """Start the audio stream."""
        try:
            self.stream = sd.InputStream(
                device=None,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
            self.stream.start()
            self._log(
                f"[STREAM] Audio stream started: {self.channels}ch, "
                f"{self.samplerate}Hz, blocksize={self.blocksize}"
            )
        except sd.PortAudioError as e:
            print(f"[ERROR] Failed to start audio stream: {e}")
            print(
                "[ERROR] Please ensure a valid input device is connected and selected."
            )

    def stop(self):
        """Stop the audio stream."""
        if hasattr(self, "stream") and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self._log("[STREAM] Audio stream stopped")

    def closeEvent(self, event):
        """Handle window close event."""
        self._log("[CLOSE] Closing LiveAudioWaveform...")
        self.stop()
        event.accept()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live Audio Waveform with LUFS Loudness Normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Run with default settings (normalization ON)
  %(prog)s -v                     # Run with verbose logging
  %(prog)s --no-norm              # Run without auto-normalization
  %(prog)s -v --no-norm           # Verbose mode, no normalization
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Disable loudness normalization (default: normalization ON)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (default: system default)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels (default: 1)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=DEFAULT_SAMPLERATE,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLERATE})",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=DEFAULT_BLOCKSIZE,
        help=f"Audio block size (default: {DEFAULT_BLOCKSIZE})",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SIZE_SECONDS,
        help=f"Window size in seconds (default: {DEFAULT_WINDOW_SIZE_SECONDS})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_AMPLITUDE_SCALE,
        help=f"Initial amplitude scale (default: {DEFAULT_AMPLITUDE_SCALE})",
    )
    return parser.parse_args()


def setup_signal_handlers(app, window):
    """Setup graceful shutdown on Ctrl+C (SIGINT)."""

    def signal_handler(signum, frame):
        print("\n[SIGNAL] Shutting down gracefully...")
        window.stop()
        QTimer.singleShot(0, app.quit)

    signal.signal(signal.SIGINT, signal_handler)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)
    return signal_handler


if __name__ == "__main__":
    args = parse_args()

    QApplication.setStyle("Fusion")
    app = QApplication(sys.argv)
    window = LiveAudioWaveform(
        device=args.device,
        channels=args.channels,
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        window_size_seconds=args.window,
        amplitude_scale=args.scale,
        enable_normalization=not args.no_norm,
        verbose=args.verbose,
    )
    setup_signal_handlers(app, window)
    window.start()
    window.show()
    print("=" * 60)
    print("Live Audio Waveform with LUFS Loudness Normalization")
    print(f"Verbose: {'ON' if args.verbose else 'OFF'}")
    print(f"Normalization: {'OFF' if args.no_norm else 'ON (auto)'}")
    print(f"Soft Clipping: ON (threshold: {SOFT_CLIP_THRESHOLD:.2f})")
    print(f"Sample Rate: {args.samplerate} Hz")
    print(f"Window Size: {args.window}s")
    print(f"Amplitude Scale: {args.scale}x")
    print("=" * 60)
    print("Features:")
    print("  ✓ LUFS-based loudness normalization (ITU-R BS.1770)")
    print("  ✓ Auto-normalization with smooth transitions")
    print("  ✓ Soft clipping to prevent visual distortion")
    print("  ✓ Peak level warnings")
    print("  ✓ Adjustable clipping threshold")
    print("  ✓ Multiple target LUFS presets")
    print("  ✓ Loudness range display")
    print("  ✓ Gain reset button")
    print("=" * 60)
    sys.exit(app.exec())

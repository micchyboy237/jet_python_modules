import argparse
import signal
import sys
import threading
from typing import Optional, Tuple

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

# =============================================================================
# Constants
# =============================================================================

# Window dimensions
OVERLAY_WIDTH = 520
OVERLAY_HEIGHT = 300

# Margins for window positioning
DEFAULT_MARGIN_RIGHT = 20
DEFAULT_MARGIN_BOTTOM = 50
DEFAULT_MARGIN_TOP = 20

# Audio defaults
DEFAULT_SAMPLERATE = 44100
DEFAULT_BLOCKSIZE = 512
DEFAULT_WINDOW_SIZE_SECONDS = 5.0

# Amplitude scaling
DEFAULT_AMPLITUDE_SCALE = 2.0
MIN_AMPLITUDE_SCALE = 0.1
MAX_AMPLITUDE_SCALE = 10.0

# Gain limits
MAX_GAIN_DB = 30.0
MIN_GAIN_DB = -20.0

# Soft clipping
DEFAULT_SOFT_CLIP_THRESHOLD = 0.95
SOFT_CLIP_MIN_THRESHOLD = 0.50
SOFT_CLIP_MAX_THRESHOLD = 0.99

# Loudness measurement
LOUDNESS_MIN_SAMPLES = 512
LOUDNESS_RANGE_CHUNK_SECONDS = 0.4
LOUDNESS_RANGE_MIN_CHUNKS = 2
LOUDNESS_SILENCE_THRESHOLD = -70.0

# Update intervals (in counter units)
LUFS_UPDATE_INTERVAL = 20
LOG_UPDATE_INTERVAL = 50
PEAK_WARNING_TIMEOUT = 10

# Auto-normalization parameters
AUTO_NORM_SMOOTHING_UP = 0.7  # Faster response for increasing gain
AUTO_NORM_SMOOTHING_DOWN = 0.9  # Slower decay for decreasing gain
AUTO_NORM_THRESHOLD_DOWN = 0.5  # Trigger if below target by this much
AUTO_NORM_THRESHOLD_UP = 1.0  # Trigger if above target by this much
AUTO_NORM_MAX_STEP = 3.0  # Maximum gain change per adjustment
AUTO_SCALE_SMOOTHING = 0.95

# RMS thresholds
RMS_MIN_SIGNAL = 0.000001
RMS_AUTO_SCALE_THRESHOLD = 0.0001

# Target LUFS presets
TARGET_LUFS_PRESETS = {
    "Streaming (-14 LUFS)": -14.0,
    "Broadcast (-23 LUFS)": -23.0,
    "YouTube (-13 LUFS)": -13.0,
    "Spotify (-14 LUFS)": -14.0,
    "Cinema (-27 LUFS)": -27.0,
}


# =============================================================================
# Circular Audio Buffer (Fixed)
# =============================================================================


class CircularAudioBuffer:
    """Memory-efficient circular buffer for audio data.

    Uses a write pointer approach to avoid np.roll() overhead.
    Provides contiguous chronological views of the data.
    """

    def __init__(self, buffer_size: int, channels: int):
        self.buffer = np.zeros((buffer_size, channels))
        self.size = buffer_size
        self.channels = channels
        self.write_pos = 0
        self.total_written = 0

    @property
    def filled(self) -> bool:
        """Check if buffer has been filled at least once."""
        return self.total_written >= self.size

    @property
    def valid_samples(self) -> int:
        """Number of valid samples currently in buffer."""
        return min(self.total_written, self.size)

    def write(self, data: np.ndarray) -> int:
        """Write new audio data to buffer. Returns number of frames written."""
        frames = data.shape[0]

        # Handle case where input is larger than buffer
        if frames > self.size:
            data = data[-self.size :]
            frames = self.size

        remaining = self.size - self.write_pos

        if frames <= remaining:
            self.buffer[self.write_pos : self.write_pos + frames] = data
            self.write_pos += frames
        else:
            # Split write across buffer boundary
            self.buffer[self.write_pos :] = data[:remaining]
            self.buffer[: frames - remaining] = data[remaining:]
            self.write_pos = frames - remaining

        self.total_written += frames

        # Keep write_pos within bounds
        if self.write_pos >= self.size:
            self.write_pos %= self.size

        return frames

    def get_contiguous_view(self, channel: int = 0) -> np.ndarray:
        """Get a contiguous chronological view of the buffer.

        Returns the data in time order (oldest first, newest last).
        """
        if not self.filled:
            # Buffer hasn't wrapped yet - return from start to write_pos
            return self.buffer[: self.write_pos, channel].copy()

        # Buffer has wrapped - concatenate to get chronological order
        return np.concatenate(
            [
                self.buffer[self.write_pos :, channel],
                self.buffer[: self.write_pos, channel],
            ]
        )

    def get_latest_frames(self, frames: int, channel: int = 0) -> np.ndarray:
        """Get the most recent N frames for a specific channel."""
        available = self.valid_samples
        if available == 0:
            return np.array([])

        frames = min(frames, available)

        if not self.filled:
            # Haven't wrapped yet
            start = max(0, self.write_pos - frames)
            return self.buffer[start : self.write_pos, channel].copy()

        # Wrapped buffer
        if self.write_pos >= frames:
            return self.buffer[self.write_pos - frames : self.write_pos, channel].copy()
        else:
            return np.concatenate(
                [
                    self.buffer[self.write_pos - frames :, channel],
                    self.buffer[: self.write_pos, channel],
                ]
            )


# =============================================================================
# Utility functions
# =============================================================================


def calculate_rms(data: np.ndarray) -> float:
    """Calculate RMS value of audio data."""
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def calculate_db(rms: float) -> float:
    """Convert RMS to decibels."""
    if rms <= RMS_MIN_SIGNAL:
        return float("-inf")
    return 20 * np.log10(rms)


def soft_clip(data: np.ndarray, threshold: float, enabled: bool = True) -> np.ndarray:
    """Apply tanh-based soft clipping."""
    if not enabled or threshold >= 1.0 or len(data) == 0:
        return data
    safe_threshold = max(threshold, 0.01)
    scaled = data / safe_threshold
    return np.tanh(scaled) * safe_threshold


def calculate_normalization_gain(
    current_lufs: float, target_lufs: float
) -> Tuple[float, float]:
    """Calculate gain needed for LUFS normalization.

    Returns:
        Tuple of (gain_db, gain_linear)
    """
    if current_lufs == float("-inf"):
        return 0.0, 1.0

    gain_db = np.clip(target_lufs - current_lufs, MIN_GAIN_DB, MAX_GAIN_DB)
    gain_linear = 10 ** (gain_db / 20.0)
    return gain_db, gain_linear


# =============================================================================
# Color mapper for level displays
# =============================================================================


class LevelColorMapper:
    """Maps audio levels to colors for consistent visual feedback."""

    @staticmethod
    def get_db_color(db_level: float) -> str:
        """Get color for dB level display."""
        if db_level == float("-inf"):
            return "#666666"
        if db_level > -3:
            return "#ff3333"
        elif db_level > -6:
            return "#ff9900"
        elif db_level > -12:
            return "#ffdd00"
        elif db_level > -20:
            return "#00ff00"
        elif db_level > -40:
            return "#00ccff"
        return "#6666ff"

    @staticmethod
    def get_lufs_color(distance: float) -> str:
        """Get color based on distance from target LUFS."""
        if distance >= -1:
            return "#00ff00"
        elif distance >= -3:
            return "#ffdd00"
        elif distance >= -6:
            return "#ff9900"
        return "#ff3333"

    @staticmethod
    def get_gain_color(gain_db: float) -> str:
        """Get color for gain display."""
        if abs(gain_db) < 0.01:
            return "#999999"
        elif gain_db > 20:
            return "#ff9900"
        elif gain_db > 12:
            return "#ffdd00"
        elif gain_db > 0:
            return "#00ff00"
        return "#ff6666"


# =============================================================================
# Main widget class
# =============================================================================


class LiveAudioWaveform(QWidget):
    """Live audio waveform visualization with LUFS-based loudness normalization."""

    audio_data_ready = pyqtSignal(np.ndarray)
    _instance_count = 0

    def __init__(
        self,
        device: Optional[int] = None,
        channels: int = 1,
        samplerate: int = DEFAULT_SAMPLERATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        window_size_seconds: float = DEFAULT_WINDOW_SIZE_SECONDS,
        amplitude_scale: float = DEFAULT_AMPLITUDE_SCALE,
        enable_normalization: bool = True,
        verbose: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        # Instance tracking
        LiveAudioWaveform._instance_count += 1
        self.instance_id = LiveAudioWaveform._instance_count

        # Configuration
        self.verbose = verbose
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize

        # Audio buffer
        buffer_size = int(window_size_seconds * samplerate)
        self.audio_buffer = CircularAudioBuffer(buffer_size, channels)

        # Display state
        self.amplitude_scale = amplitude_scale
        self.auto_scale = False
        self.soft_clip_enabled = True
        self.soft_clip_threshold = DEFAULT_SOFT_CLIP_THRESHOLD

        # Audio processing state
        self.current_rms = 0.0
        self.current_lufs = float("-inf")
        self.gain_db = 0.0
        self.gain_linear = 1.0
        self.clip_count = 0
        self.peak_warning_active = False
        self.last_peak_warning = 0
        self.loudness_range = 0.0

        # Thread safety for gain
        self._gain_lock = threading.Lock()

        # Loudness measurement
        self.target_lufs = -14.0
        self.loudness_normalization_enabled = enable_normalization
        self.loudness_meter = pyln.Meter(samplerate)

        # Update tracking
        self.update_counter = 0

        # Color mapper
        self.colors = LevelColorMapper()

        # Window setup
        self._setup_window()
        self._setup_ui()

        # Connect signals
        self.audio_data_ready.connect(self.update_plot)

        # Initialize normalization if enabled
        if enable_normalization:
            self.auto_normalize_cb.setChecked(True)

        self._log(
            f"[INIT #{self.instance_id}] LiveAudioWaveform initialized | "
            f"scale={amplitude_scale:.1f}x | target={self.target_lufs:.0f} LUFS | "
            f"normalization={'ON' if enable_normalization else 'OFF'} | "
            f"buffer={buffer_size} samples ({window_size_seconds}s)"
        )

    # =========================================================================
    # Thread-safe gain access
    # =========================================================================

    def _get_gain_linear(self) -> float:
        """Thread-safe gain read."""
        with self._gain_lock:
            return self.gain_linear

    def _get_gain_db(self) -> float:
        """Thread-safe gain db read."""
        with self._gain_lock:
            return self.gain_db

    def _set_gain(self, gain_db: float):
        """Thread-safe gain update."""
        gain_db = np.clip(gain_db, MIN_GAIN_DB, MAX_GAIN_DB)
        with self._gain_lock:
            self.gain_db = gain_db
            self.gain_linear = 10 ** (gain_db / 20.0)

    # =========================================================================
    # Window and UI setup
    # =========================================================================

    def _setup_window(self):
        """Configure window properties."""
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

        self.drag_position = None

    def _log(self, message: str):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.instance_id}] {message}")

    def _setup_ui(self):
        """Set up the complete user interface."""
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
        """Create the title bar."""
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
        """Create controls for waveform height adjustment."""
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
        self.amp_slider.setStyleSheet(self._get_slider_style("#00ffff"))
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
        self.auto_scale_cb.setStyleSheet(self._get_checkbox_style("#00ffff"))
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

        self.auto_normalize_cb = QCheckBox("Auto")
        self.auto_normalize_cb.setStyleSheet(self._get_checkbox_style("#00aa44"))
        self.auto_normalize_cb.toggled.connect(self._on_auto_normalize_toggled)
        self.auto_normalize_cb.setToolTip(
            "Automatically adjust gain when audio is soft"
        )
        loudness_layout.addWidget(self.auto_normalize_cb)

        self.lufs_label = QLabel("LUFS: -∞")
        self.lufs_label.setStyleSheet("color: #999999; font-size: 10px;")
        loudness_layout.addWidget(self.lufs_label)

        self.gain_label = QLabel("Gain: +0.0 dB")
        self.gain_label.setStyleSheet("color: #999999; font-size: 10px;")
        loudness_layout.addWidget(self.gain_label)

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

        self.soft_clip_cb = QCheckBox("Soft Clip")
        self.soft_clip_cb.setChecked(self.soft_clip_enabled)
        self.soft_clip_cb.setStyleSheet(self._get_checkbox_style("#ff9900"))
        self.soft_clip_cb.toggled.connect(self._on_soft_clip_toggled)
        self.soft_clip_cb.setToolTip("Apply soft clipping to prevent visual distortion")
        clipping_layout.addWidget(self.soft_clip_cb)

        self.clip_count_label = QLabel("")
        self.clip_count_label.setStyleSheet("color: #777777; font-size: 9px;")
        clipping_layout.addWidget(self.clip_count_label)

        threshold_label = QLabel("Thresh:")
        threshold_label.setStyleSheet("color: #888888; font-size: 9px;")
        clipping_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(
            int(SOFT_CLIP_MIN_THRESHOLD * 100), int(SOFT_CLIP_MAX_THRESHOLD * 100)
        )
        self.threshold_slider.setValue(int(self.soft_clip_threshold * 100))
        self.threshold_slider.setFixedWidth(80)
        self.threshold_slider.setStyleSheet(self._get_slider_style("#ff9900"))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        clipping_layout.addWidget(self.threshold_slider)

        self.threshold_label = QLabel(f"{self.soft_clip_threshold:.2f}")
        self.threshold_label.setStyleSheet(
            "color: #ff9900; font-size: 9px; font-weight: bold;"
        )
        clipping_layout.addWidget(self.threshold_label)

        clipping_layout.addStretch()
        self.main_layout.addWidget(self.clipping_bar)
        self._log("[UI] Clipping bar created")

    def _create_plot_widget(self):
        """Create the waveform plot widget."""
        self.plot_widget = pg.PlotWidget(title="")
        self.plot_widget.setBackground("#1e1e1e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getPlotItem().vb.setMouseEnabled(x=True, y=False)
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)

        self._update_y_range()

        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")

        self.curve = self.plot_widget.plot(pen=pg.mkPen("#00ffff", width=2.5))

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
        self._log("[UI] Plot widget created")

    def _add_reference_lines(self):
        """Add horizontal reference lines."""
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

    # =========================================================================
    # Style helpers
    # =========================================================================

    @staticmethod
    def _get_slider_style(color: str) -> str:
        """Generate slider stylesheet."""
        return (
            f"QSlider::groove:horizontal {{ height: 6px; background: #555; "
            f"border-radius: 3px; }} "
            f"QSlider::handle:horizontal {{ background: {color}; width: 16px; "
            f"margin: -5px 0; border-radius: 8px; }} "
            f"QSlider::sub-page:horizontal {{ background: {color}; border-radius: 3px; }}"
        )

    @staticmethod
    def _get_checkbox_style(color: str) -> str:
        """Generate checkbox stylesheet."""
        return (
            "QCheckBox { color: #cccccc; font-size: 11px; font-weight: bold; "
            "spacing: 4px; } "
            "QCheckBox::indicator { width: 16px; height: 16px; } "
            "QCheckBox::indicator:unchecked { background: #555; border: 2px solid #666; "
            "border-radius: 3px; } "
            f"QCheckBox::indicator:checked {{ background: {color}; "
            "border-radius: 3px; }"
        )

    # =========================================================================
    # Display updates
    # =========================================================================

    def _update_y_range(self):
        """Update the Y-axis range and reference lines."""
        y_max = self.amplitude_scale * 1.1
        self.plot_widget.setYRange(-y_max, y_max)
        self.plot_widget.getPlotItem().enableAutoRange(axis="y", enable=False)
        self._update_reference_lines()

        if hasattr(self, "scale_label"):
            self.scale_label.setText(f"Scale: {self.amplitude_scale:.1f}x")

        self._log(
            f"[RANGE] Y-axis: [-{y_max:.1f}, {y_max:.1f}] | scale={self.amplitude_scale:.1f}x"
        )

    def _update_reference_lines(self):
        """Update reference line positions."""
        if hasattr(self, "top_line"):
            self.top_line.setPos(self.amplitude_scale)
            self.bottom_line.setPos(-self.amplitude_scale)

        if hasattr(self, "clip_line_top"):
            clip_pos = self.soft_clip_threshold * self.amplitude_scale
            self.clip_line_top.setPos(clip_pos)
            self.clip_line_bottom.setPos(-clip_pos)

    def _update_rms_display(self, raw_rms: float):
        """Update RMS level display."""
        db_level = calculate_db(raw_rms)

        if db_level == float("-inf"):
            self.rms_label.setText("Level: -∞ dB")
            self.rms_label.setStyleSheet("color: #666666; font-size: 10px;")
            return

        gain_linear = self._get_gain_linear()
        gained_db = calculate_db(raw_rms * gain_linear)

        gain_db = self._get_gain_db()
        if abs(gain_db) > 0.1:
            self.rms_label.setText(f"Level: {gained_db:.1f} dB (raw: {db_level:.1f})")
        else:
            self.rms_label.setText(f"Level: {db_level:.1f} dB")

        color = self.colors.get_db_color(gained_db)
        self.rms_label.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold;"
        )

    def _update_lufs_display(self, lufs: float):
        """Update LUFS display."""
        if lufs == float("-inf"):
            self.lufs_label.setText("LUFS: -∞")
            self.lufs_label.setStyleSheet("color: #666666; font-size: 10px;")
            return

        gain_db = self._get_gain_db()
        effective_lufs = lufs + gain_db
        self.lufs_label.setText(f"LUFS: {effective_lufs:.1f}")

        distance = effective_lufs - self.target_lufs
        color = self.colors.get_lufs_color(distance)
        self.lufs_label.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold;"
        )

    def _update_gain_display(self):
        """Update gain display label."""
        gain_db = self._get_gain_db()

        if abs(gain_db) < 0.01:
            self.gain_label.setText("Gain: +0.0 dB")
        elif gain_db > 0:
            self.gain_label.setText(f"Gain: +{gain_db:.1f} dB")
        else:
            self.gain_label.setText(f"Gain: {gain_db:.1f} dB")

        color = self.colors.get_gain_color(gain_db)
        self.gain_label.setStyleSheet(
            f"color: {color}; font-size: 10px; font-weight: bold;"
        )

    def _check_peaks(self, data: np.ndarray):
        """Check for peaks and update warning display."""
        if len(data) == 0:
            return

        max_peak = np.max(np.abs(data))

        if max_peak > 0.95:
            self.clip_count += 1

            if not self.peak_warning_active:
                self.peak_warning_active = True
                self._log(f"[WARNING] Digital clipping detected! Peak: {max_peak:.3f}")

            self.peak_warning_label.setText(f"⚠ PEAK: {max_peak:.2f}")
            self.peak_warning_label.setStyleSheet(
                "color: #ff3333; font-size: 11px; font-weight: bold;"
            )
            self.last_peak_warning = self.update_counter

            self.clip_count_label.setText(f"Clips: {self.clip_count}")
            self.clip_count_label.setStyleSheet(
                "color: #ff3333; font-size: 9px; font-weight: bold;"
            )
        elif (
            self.peak_warning_active
            and (self.update_counter - self.last_peak_warning) > PEAK_WARNING_TIMEOUT
        ):
            self.peak_warning_active = False
            self.peak_warning_label.setText("")
            self.clip_count_label.setStyleSheet("color: #777777; font-size: 9px;")

    def _measure_loudness(self, audio_data: np.ndarray) -> float:
        """Measure integrated loudness using pyloudnorm.

        Handles the case where audio_data is shorter than required by pyloudnorm.
        Suppresses errors during initial buffer filling.
        """
        if len(audio_data) < LOUDNESS_MIN_SAMPLES:
            return float("-inf")

        try:
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            loudness = self.loudness_meter.integrated_loudness(audio_data)
            return float(loudness)
        except Exception as e:
            # Only log if we've been running for a while (avoid spam during startup)
            if self.update_counter > LUFS_UPDATE_INTERVAL * 2:
                self._log(f"[LOUDNESS] Measurement error: {e}")
            return float("-inf")

    def _calculate_loudness_range(self, audio_data: np.ndarray) -> float:
        """Calculate short-term loudness range."""
        chunk_samples = int(self.samplerate * LOUDNESS_RANGE_CHUNK_SECONDS)

        if len(audio_data) < chunk_samples * LOUDNESS_RANGE_MIN_CHUNKS:
            return 0.0

        try:
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            num_chunks = len(audio_data) // chunk_samples
            if num_chunks < LOUDNESS_RANGE_MIN_CHUNKS:
                return 0.0

            loudness_values = []
            for i in range(num_chunks):
                chunk = audio_data[i * chunk_samples : (i + 1) * chunk_samples]
                try:
                    lufs = self.loudness_meter.integrated_loudness(chunk)
                    if lufs > LOUDNESS_SILENCE_THRESHOLD:
                        loudness_values.append(lufs)
                except Exception:
                    pass

            if len(loudness_values) < LOUDNESS_RANGE_MIN_CHUNKS:
                return 0.0

            return max(loudness_values) - min(loudness_values)
        except Exception as e:
            self._log(f"[LOUDNESS] Range calculation error: {e}")
            return 0.0

    # =========================================================================
    # Control event handlers
    # =========================================================================

    def _on_amplitude_slider_changed(self, value: int):
        """Handle amplitude slider changes."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:
            self.amplitude_scale = scale

            self.amp_spinbox.blockSignals(True)
            self.amp_spinbox.setValue(value)
            self.amp_spinbox.blockSignals(False)

            self._update_y_range()
            self._log(f"[CONTROL] Amplitude scale: {scale:.2f}x (slider)")

    def _on_amplitude_spinbox_changed(self, value: int):
        """Handle amplitude spinbox changes."""
        scale = value / 100.0
        if abs(scale - self.amplitude_scale) > 0.001:
            self.amplitude_scale = scale

            self.amp_slider.blockSignals(True)
            self.amp_slider.setValue(value)
            self.amp_slider.blockSignals(False)

            self._update_y_range()
            self._log(f"[CONTROL] Amplitude scale: {scale:.2f}x (spinbox)")

    def _on_auto_scale_toggled(self, checked: bool):
        """Handle auto-scale toggle."""
        self.auto_scale = checked
        self.amp_slider.setEnabled(not checked)
        self.amp_spinbox.setEnabled(not checked)

        if not checked:
            self.amplitude_scale = self.amp_slider.value() / 100.0
            self._update_y_range()

        self._log(f"[AUTO] Auto-scale {'ON' if checked else 'OFF'}")

    def _on_target_changed(self, text: str):
        """Handle target LUFS preset change."""
        self.target_lufs = TARGET_LUFS_PRESETS[text]
        self._log(f"[LOUDNESS] Target: {text} = {self.target_lufs:.0f} LUFS")

        if self.loudness_normalization_enabled:
            self._on_normalize_clicked()

    def _on_auto_normalize_toggled(self, checked: bool):
        """Handle auto-normalize toggle."""
        self.loudness_normalization_enabled = checked
        self._log(f"[LOUDNESS] Auto-normalization {'ON' if checked else 'OFF'}")

        if checked:
            self._update_gain_display()

    def _on_soft_clip_toggled(self, checked: bool):
        """Handle soft clip toggle."""
        self.soft_clip_enabled = checked
        self._log(f"[CLIP] Soft clipping {'ON' if checked else 'OFF'}")

    def _on_threshold_changed(self, value: int):
        """Handle soft clip threshold change."""
        self.soft_clip_threshold = value / 100.0
        self.threshold_label.setText(f"{self.soft_clip_threshold:.2f}")
        self._update_reference_lines()
        self._log(f"[CLIP] Threshold: {self.soft_clip_threshold:.2f}")

    def _on_normalize_clicked(self):
        """Handle manual normalization."""
        self._log("[LOUDNESS] Manual normalization triggered")

        display_data = self.audio_buffer.get_contiguous_view(0)
        current_lufs = self._measure_loudness(display_data)

        if current_lufs == float("-inf"):
            self._log("[LOUDNESS] Cannot normalize - insufficient audio data")
            return

        gain_db, gain_linear = calculate_normalization_gain(
            current_lufs, self.target_lufs
        )

        if abs(gain_db) < 0.5:
            self._log("[LOUDNESS] Already near target level")
            self._set_gain(0.0)
            self._update_gain_display()
            return

        self._set_gain(gain_db)
        self._update_gain_display()

        self.loudness_range = self._calculate_loudness_range(display_data)
        self.range_label.setText(f"Range: {self.loudness_range:.1f} LU")

        self._log(f"[LOUDNESS] Applied {gain_db:+.1f} dB gain ({gain_linear:.2f}x)")

    def _on_reset_gain_clicked(self):
        """Reset all gain and clipping state."""
        self._log("[LOUDNESS] Gain reset to 0 dB")

        self._set_gain(0.0)
        self.clip_count = 0
        self.peak_warning_active = False

        self.peak_warning_label.setText("")
        self.clip_count_label.setText("")
        self._update_gain_display()
        self.range_label.setText("")

    # =========================================================================
    # Mouse events for window dragging
    # =========================================================================

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for window dragging."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        control_areas = [
            self.close_btn.geometry(),
            self.controls_bar.geometry(),
        ]

        if hasattr(self, "loudness_bar"):
            control_areas.append(self.loudness_bar.geometry())
        if hasattr(self, "clipping_bar"):
            control_areas.append(self.clipping_bar.geometry())

        for area in control_areas:
            if area.contains(event.pos()):
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

    # =========================================================================
    # Audio processing
    # =========================================================================

    def audio_callback(
        self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags
    ):
        """Audio stream callback (runs in background thread).

        Must be fast and non-blocking.
        """
        if status:
            self._log(f"[AUDIO] Status: {status}")

        self.audio_data_ready.emit(indata.copy())

    def update_plot(self, new_data: np.ndarray):
        """Main update loop (runs in GUI thread)."""
        self.update_counter += 1

        # Write to circular buffer
        new_frames = self.audio_buffer.write(new_data)

        # Get latest frames for RMS calculation (fast)
        latest_chunk = self.audio_buffer.get_latest_frames(new_frames, 0)
        raw_rms = calculate_rms(latest_chunk)
        self.current_rms = raw_rms
        self._update_rms_display(raw_rms)

        # Get full display data for waveform
        display_data = self.audio_buffer.get_contiguous_view(0)

        # Apply gain (thread-safe)
        gain_linear = self._get_gain_linear()
        gained_data = display_data * gain_linear

        # Check peaks on gained data
        self._check_peaks(gained_data)

        # Apply soft clipping
        gained_data = soft_clip(
            gained_data, self.soft_clip_threshold, self.soft_clip_enabled
        )

        # Periodic LUFS measurement (expensive, batched)
        if self.update_counter % LUFS_UPDATE_INTERVAL == 0:
            self._process_loudness(display_data)

        # Auto-scale logic
        if self.auto_scale and raw_rms > RMS_AUTO_SCALE_THRESHOLD:
            self._apply_auto_scale(raw_rms, gain_linear)

        # Scale for display and plot
        scaled_data = gained_data * self.amplitude_scale
        self.curve.setData(scaled_data)

        # Periodic detailed logging
        if self.update_counter % LOG_UPDATE_INTERVAL == 0:
            self._log_detailed_stats(raw_rms, gain_linear, scaled_data)

    def _process_loudness(self, display_data: np.ndarray):
        """Process loudness measurement and auto-normalization."""
        # Measure loudness on raw (pre-gain) data
        measured_lufs = self._measure_loudness(display_data)
        self.current_lufs = measured_lufs
        self._update_lufs_display(measured_lufs)

        if not self.loudness_normalization_enabled:
            return

        if measured_lufs == float("-inf"):
            return

        current_gain_db = self._get_gain_db()
        effective_lufs = measured_lufs + current_gain_db

        # Check if adjustment needed (below target)
        if effective_lufs < self.target_lufs - AUTO_NORM_THRESHOLD_DOWN:
            gain_needed = self.target_lufs - measured_lufs
            gain_needed = np.clip(gain_needed, MIN_GAIN_DB, MAX_GAIN_DB)

            # Limit step size to prevent overshoot
            max_step = current_gain_db + AUTO_NORM_MAX_STEP
            gain_needed = min(gain_needed, max_step)

            # Smooth upward adjustment
            new_gain = current_gain_db * AUTO_NORM_SMOOTHING_UP + gain_needed * (
                1 - AUTO_NORM_SMOOTHING_UP
            )
            new_gain = np.clip(new_gain, MIN_GAIN_DB, MAX_GAIN_DB)

            self._set_gain(new_gain)
            self._update_gain_display()

            self._log(
                f"[AUTO-NORM] Adjusting up: raw={measured_lufs:.1f} LUFS, "
                f"eff={effective_lufs:.1f}, target={self.target_lufs:.0f}, "
                f"gain={new_gain:.1f} dB"
            )

        # Check if adjustment needed (above target + hysteresis)
        elif effective_lufs > self.target_lufs + AUTO_NORM_THRESHOLD_UP:
            # Smooth downward adjustment
            new_gain = current_gain_db * AUTO_NORM_SMOOTHING_DOWN
            new_gain = np.clip(new_gain, MIN_GAIN_DB, MAX_GAIN_DB)

            self._set_gain(new_gain)
            self._update_gain_display()

            self._log(
                f"[AUTO-NORM] Adjusting down: raw={measured_lufs:.1f} LUFS, "
                f"eff={effective_lufs:.1f}, target={self.target_lufs:.0f}, "
                f"gain={new_gain:.1f} dB"
            )

    def _apply_auto_scale(self, raw_rms: float, gain_linear: float):
        """Apply automatic amplitude scaling."""
        effective_rms = raw_rms * gain_linear
        target_scale = (self.amplitude_scale * 0.7) / max(
            effective_rms, RMS_AUTO_SCALE_THRESHOLD
        )

        # Smooth scale transition
        self.amplitude_scale = (
            self.amplitude_scale * AUTO_SCALE_SMOOTHING
            + target_scale * (1 - AUTO_SCALE_SMOOTHING)
        )
        self.amplitude_scale = float(
            np.clip(self.amplitude_scale, MIN_AMPLITUDE_SCALE, MAX_AMPLITUDE_SCALE)
        )
        self._update_y_range()

    def _log_detailed_stats(
        self, raw_rms: float, gain_linear: float, scaled_data: np.ndarray
    ):
        """Log detailed statistics for debugging."""
        gain_db = self._get_gain_db()
        self._log(
            f"[PLOT] Update #{self.update_counter}: "
            f"scale={self.amplitude_scale:.2f}x | "
            f"gain={gain_db:.1f}dB | "
            f"raw RMS={raw_rms:.4f} | "
            f"eff RMS={raw_rms * gain_linear:.4f} | "
            f"raw LUFS={self.current_lufs:.1f} | "
            f"eff LUFS={self.current_lufs + gain_db:.1f} | "
            f"peak={np.max(np.abs(scaled_data)):.3f} | "
            f"clips={self.clip_count}"
        )

    # =========================================================================
    # Stream control
    # =========================================================================

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
                f"[STREAM] Started: {self.channels}ch, {self.samplerate}Hz, "
                f"blocksize={self.blocksize}"
            )
        except sd.PortAudioError as e:
            print(f"[ERROR] Failed to start audio stream: {e}")
            print("[ERROR] Please ensure a valid input device is connected.")

    def stop(self):
        """Stop the audio stream."""
        if hasattr(self, "stream") and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self._log("[STREAM] Audio stream stopped")

    def closeEvent(self, event):
        """Handle window close event."""
        self._log(f"[CLOSE #{self.instance_id}] Shutting down...")
        self.stop()
        event.accept()


# =============================================================================
# Command-line interface
# =============================================================================


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
        "-v", "--verbose", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument(
        "--no-norm", action="store_true", help="Disable loudness normalization"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
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


def setup_signal_handlers(app: QApplication, window: LiveAudioWaveform):
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


def print_startup_info(args):
    """Print startup information banner."""
    print("=" * 60)
    print("Live Audio Waveform with LUFS Loudness Normalization")
    print(f"Verbose: {'ON' if args.verbose else 'OFF'}")
    print(f"Normalization: {'OFF' if args.no_norm else 'ON (auto)'}")
    print(f"Soft Clipping: ON (threshold: {DEFAULT_SOFT_CLIP_THRESHOLD:.2f})")
    print(f"Sample Rate: {args.samplerate} Hz")
    print(f"Window Size: {args.window}s")
    print(f"Amplitude Scale: {args.scale}x")
    print("=" * 60)
    print("Features:")
    print("  ✓ LUFS-based loudness normalization (ITU-R BS.1770)")
    print("  ✓ Auto-normalization with bounded step sizes")
    print("  ✓ Soft clipping to prevent visual distortion")
    print("  ✓ Peak level warnings")
    print("  ✓ Adjustable clipping threshold")
    print("  ✓ Multiple target LUFS presets")
    print("  ✓ Loudness range display")
    print("  ✓ Gain reset button")
    print("  ✓ Optimized circular buffer (no np.roll)")
    print("  ✓ Thread-safe gain management")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

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

    print_startup_info(args)

    sys.exit(app.exec())

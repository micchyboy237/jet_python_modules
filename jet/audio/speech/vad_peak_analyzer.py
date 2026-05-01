import logging
from typing import Any, Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


class VADSegment(TypedDict):
    frame_start: int  # Starting frame index (inclusive)
    frame_end: int  # Ending frame index (inclusive)
    frame_length: int  # Number of frames
    start_s: float  # Start time in seconds
    end_s: float  # End time in seconds
    duration_s: float  # Duration in seconds
    details: Dict[str, Any]  # Additional insights (peak/trough properties)


class VADPeakAnalyzer:
    """
    Analyzes peaks (local maxima) and troughs (local minima) in VAD speech probabilities.
    Enhanced with optional debug logging for diagnostics.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 32.0,
        debug: bool = False,
    ):
        """
        Args:
            sample_rate: Audio sample rate in Hz.
            frame_duration_ms: Duration of each VAD frame in milliseconds.
            debug: If True, enable debug logging.
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_duration_s = frame_duration_ms / 1000.0
        self.hop_length = int(sample_rate * self.frame_duration_s)  # samples per frame
        self.debug = debug

        if debug:
            logging.basicConfig(
                level=logging.DEBUG, format="%(levelname)s - %(message)s"
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.WARNING)

    def _log_debug(self, msg: str, **kwargs):
        if self.debug:
            self.logger.debug(msg, extra=kwargs)

    def _compute_times(self, frame_idx: int) -> tuple[float, float]:
        """Convert frame index to start/end time in seconds."""
        start_s = frame_idx * self.frame_duration_s
        end_s = (frame_idx + 1) * self.frame_duration_s
        return start_s, end_s

    def extract_peaks(
        self,
        probs: List[float],
        height: Optional[float] = None,
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[VADSegment]:
        """
        Extract peaks (local maxima) from VAD probabilities.

        Recommended params for speech VAD (tune based on your model):
        - height: min probability (e.g. 0.6)
        - distance: min frames between peaks (e.g. 5-20)
        - prominence: how much it stands out (e.g. 0.1-0.3)
        """
        if not probs:
            return []

        x = np.array(probs, dtype=float)
        self._log_debug(
            f"extract_peaks called with height={height}, distance={distance}, prominence={prominence}"
        )
        self._log_debug(f"Input probs: {[round(p, 4) for p in probs]}")

        peaks_idx, properties = find_peaks(
            x,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width,
            **kwargs,
        )

        self._log_debug(f"Raw peaks found at indices: {peaks_idx.tolist()}")
        if len(peaks_idx) > 0:
            self._log_debug(
                f"Peak probabilities: {[round(x[i], 4) for i in peaks_idx]}"
            )
            if "prominences" in properties:
                self._log_debug(
                    f"Prominences: {[round(p, 4) for p in properties['prominences']]}"
                )
            if "left_bases" in properties and "right_bases" in properties:
                for i, idx in enumerate(peaks_idx):
                    left = properties["left_bases"][i]
                    right = properties["right_bases"][i]
                    self._log_debug(
                        f"Peak at {idx}: left_base={left}, right_base={right}, base_range=[{left}:{right + 1}]"
                    )

        segments: List[VADSegment] = []
        for i, peak in enumerate(peaks_idx):
            frame_start = int(peak)
            frame_end = int(peak)

            start_s, end_s = self._compute_times(frame_start)
            duration_s = end_s - start_s

            details = {
                "peak_index": int(peak),
                "peak_probability": float(x[peak]),
                "prominence": float(properties.get("prominences", [0])[i])
                if "prominences" in properties
                else None,
                "width": float(properties.get("widths", [0])[i])
                if "widths" in properties
                else None,
                "left_base": int(properties.get("left_bases", [0])[i])
                if "left_bases" in properties
                else None,
                "right_base": int(properties.get("right_bases", [0])[i])
                if "right_bases" in properties
                else None,
            }

            segments.append(
                {
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "frame_length": 1,
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                    "duration_s": round(duration_s, 4),
                    "details": details,
                }
            )

        self._log_debug(f"Returning {len(segments)} peak segments")
        return segments

    def extract_troughs(
        self,
        probs: List[float],
        height: Optional[
            float
        ] = None,  # For troughs, this acts as max height (use negative for find_peaks)
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[VADSegment]:
        """
        Extract troughs (local minima) by finding peaks on the negated signal.
        """
        if not probs:
            return []

        x = np.array(probs, dtype=float)
        self._log_debug(
            f"extract_troughs called with height={height}, distance={distance}, prominence={prominence}"
        )
        self._log_debug(f"Input probs: {[round(p, 4) for p in probs]}")

        # Negate to turn minima into maxima
        troughs_idx, properties = find_peaks(
            -x,
            height=-height if height is not None else None,  # invert threshold
            distance=distance,
            prominence=prominence,
            width=width,
            **kwargs,
        )

        self._log_debug(f"Raw troughs found at indices: {troughs_idx.tolist()}")
        if len(troughs_idx) > 0:
            self._log_debug(
                f"Trough probabilities: {[round(x[i], 4) for i in troughs_idx]}"
            )
            if "prominences" in properties:
                self._log_debug(
                    f"Prominences: {[round(p, 4) for p in properties['prominences']]}"
                )
            # No left/right base for troughs unless needed

        segments: List[VADSegment] = []
        for i, trough in enumerate(troughs_idx):
            frame_start = int(trough)
            frame_end = int(trough)

            start_s, end_s = self._compute_times(frame_start)
            duration_s = end_s - start_s

            details = {
                "trough_index": int(trough),
                "trough_probability": float(x[trough]),
                "prominence": float(properties.get("prominences", [0])[i])
                if "prominences" in properties
                else None,
                "width": float(properties.get("widths", [0])[i])
                if "widths" in properties
                else None,
            }

            segments.append(
                {
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "frame_length": 1,
                    "start_s": round(start_s, 4),
                    "end_s": round(end_s, 4),
                    "duration_s": round(duration_s, 4),
                    "details": details,
                }
            )

        self._log_debug(f"Returning {len(segments)} trough segments")
        return segments

    def save_plot(
        self,
        probs: List[float],
        peaks: List[VADSegment],
        troughs: List[VADSegment],
        output_path: str = "vad_peaks_troughs.png",
        title: str = "VAD Probability - Peaks and Troughs",
    ) -> None:
        """
        Save a visualization plot highlighting peaks and troughs.

        Args:
            probs: Original list of VAD probabilities.
            peaks: List of peak segments returned by extract_peaks().
            troughs: List of trough segments returned by extract_troughs().
            output_path: Path where the plot image will be saved.
            title: Title of the plot.
        """
        if not probs:
            self._log_debug("Cannot plot: empty probability list")
            return

        x = np.array(probs, dtype=float)
        frames = np.arange(len(x))

        plt.figure(figsize=(14, 7))
        plt.plot(frames, x, "b-", linewidth=2, label="VAD Probability", alpha=0.8)

        # Plot peaks
        if peaks:
            peak_indices = [p["frame_start"] for p in peaks]
            peak_probs = [p["details"]["peak_probability"] for p in peaks]
            plt.plot(
                peak_indices, peak_probs, "go", markersize=10, label="Peaks (Speech)"
            )
            for idx, prob in zip(peak_indices, peak_probs):
                plt.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="green",
                )

        # Plot troughs
        if troughs:
            trough_indices = [t["frame_start"] for t in troughs]
            trough_probs = [t["details"]["trough_probability"] for t in troughs]
            plt.plot(
                trough_indices,
                trough_probs,
                "ro",
                markersize=10,
                label="Troughs (Silence)",
            )
            for idx, prob in zip(trough_indices, trough_probs):
                plt.annotate(
                    f"{prob:.2f}",
                    xy=(idx, prob),
                    xytext=(0, -18),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="red",
                )

        plt.title(title, fontsize=16)
        plt.xlabel("Frame Index")
        plt.ylabel("Speech Probability")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self._log_debug(f"Plot saved successfully to: {output_path}")
        print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    import json
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Example: 32ms frames at 16kHz
    analyzer = VADPeakAnalyzer(sample_rate=16000, frame_duration_ms=32.0)

    probs = [
        0.1,
        0.15,
        0.8,
        0.92,
        0.85,
        0.3,
        0.12,
        0.05,
        0.88,
        0.95,
        0.7,
        0.2,
    ]  # sample VAD probs

    peaks = analyzer.extract_peaks(probs, height=0.7, prominence=0.1, distance=3)

    troughs = analyzer.extract_troughs(
        probs,
        height=0.3,  # max prob for a trough
        prominence=0.15,
        distance=3,
    )

    print("Peaks:", peaks)
    print("Troughs:", troughs)

    # Save visualization
    analyzer.save_plot(
        probs, peaks, troughs, output_path=str(OUTPUT_DIR / "vad_analysis_plot.png")
    )

    # Save peaks and troughs as JSON, and print full saved paths
    peaks_path = OUTPUT_DIR / "peaks.json"
    with open(peaks_path, "w", encoding="utf-8") as f:
        json.dump(peaks, f, ensure_ascii=False, indent=2)
    print(f"Peaks saved to: {peaks_path.resolve()}")

    troughs_path = OUTPUT_DIR / "troughs.json"
    with open(troughs_path, "w", encoding="utf-8") as f:
        json.dump(troughs, f, ensure_ascii=False, indent=2)
    print(f"Troughs saved to: {troughs_path.resolve()}")

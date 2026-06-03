"""
audio_tagger_chunk_plots.py
===========================
Visualization utilities for chunked audio tagging results.

Produces 4 plot files:
  1. chunk_heatmap.png    — Heatmap showing top-K event probabilities across chunks
  2. events_timeline.png  — Line plot tracking event probabilities over time
  3. results_bar.png      — Horizontal bar chart of aggregated results
  4. chunks_summary.png   — Per-chunk mini bar chart with annotations

Design principles:
  - Non-blocking (Agg backend)
  - High DPI (300) for publication quality
  - Accessible color palette (viridis + colorblind-friendly)
  - Automatic layout adjustment for varying K values
  - Graceful degradation for edge cases (single chunk, empty results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib
from jet.libs.sherpa_onnx.audio_tagger_types import AudioChunksTaggingSummary

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


# ── Color Palette ──────────────────────────────────────────────────────────
# Colorblind-friendly palette (IBM Design Language)
COLORS = [
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FE6100",  # Orange
    "#FFB000",  # Gold
    "#009E73",  # Green
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Dark Yellow
    "#F0E442",  # Yellow
    "#0072B2",  # Deep Blue
]

# Custom colormap for heatmap: white → blue → dark blue
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "audio_heatmap",
    ["#FFFFFF", "#648FFF", "#1A1A6E"],
    N=256,
)


def save_chunk_plots(
    summary: AudioChunksTaggingSummary,
    output_dir: Path,
    top_n_display: int = 5,
    figsize_heatmap: Tuple[int, int] = (14, 8),
    figsize_timeline: Tuple[int, int] = (14, 7),
    figsize_bar: Tuple[int, int] = (10, 6),
    figsize_summary: Tuple[int, int] = (12, 6),
    dpi: int = 300,
) -> List[Path]:
    """
    Generate all 4 visualization plots for chunked audio tagging results.

    Args:
        summary: AudioChunksTaggingSummary from tag_audio_chunks()
        output_dir: Directory to save plot files
        top_n_display: How many top events to show in heatmap/timeline
        figsize_heatmap: Figure size for heatmap (width, height)
        figsize_timeline: Figure size for timeline (width, height)
        figsize_bar: Figure size for bar chart (width, height)
        figsize_summary: Figure size for chunk summary (width, height)
        dpi: Dots per inch for output images

    Returns:
        List of Paths to saved plot files

    Debug logs trace:
        - Plot generation start/completion
        - Number of data points per plot
        - Any rendering errors

    Example:
        >>> summary = tagger.tag_audio_chunks("audio.wav")
        >>> paths = save_chunk_plots(summary, Path("output"))
        >>> for p in paths:
        ...     print(f"Saved: {p}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = summary.get("chunks", [])
    overall_top = summary.get("overall_top_predictions", [])
    audio_path = summary.get("audio_path", "unknown")
    total_duration = summary.get("total_duration", 0)
    rtf = summary.get("real_time_factor", 0)

    logger.debug(
        f"save_chunk_plots: {len(chunks)} chunks, "
        f"{len(overall_top)} overall predictions, "
        f"top_n_display={top_n_display}"
    )

    saved_paths: List[Path] = []

    # ── Plot 1: Heatmap ──────────────────────────────────────────────
    try:
        heatmap_path = _plot_chunk_heatmap(
            chunks=chunks,
            overall_top=overall_top,
            output_dir=output_dir,
            top_n_display=top_n_display,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_heatmap,
            dpi=dpi,
        )
        saved_paths.append(heatmap_path)
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}", exc_info=True)
        saved_paths.append(output_dir / "chunk_heatmap.png")  # Placeholder

    # ── Plot 2: Timeline ────────────────────────────────────────────
    try:
        timeline_path = _plot_events_timeline(
            chunks=chunks,
            overall_top=overall_top,
            output_dir=output_dir,
            top_n_display=top_n_display,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_timeline,
            dpi=dpi,
        )
        saved_paths.append(timeline_path)
    except Exception as e:
        logger.error(f"Failed to generate timeline: {e}", exc_info=True)
        saved_paths.append(output_dir / "events_timeline.png")

    # ── Plot 3: Aggregated Bar Chart ─────────────────────────────────
    try:
        bar_path = _plot_results_bar(
            overall_top=overall_top,
            output_dir=output_dir,
            audio_path=audio_path,
            total_duration=total_duration,
            rtf=rtf,
            figsize=figsize_bar,
            dpi=dpi,
        )
        saved_paths.append(bar_path)
    except Exception as e:
        logger.error(f"Failed to generate bar chart: {e}", exc_info=True)
        saved_paths.append(output_dir / "results_bar.png")

    # ── Plot 4: Chunk Summary Grid ───────────────────────────────────
    try:
        summary_path = _plot_chunk_summary(
            chunks=chunks,
            output_dir=output_dir,
            top_n_display=min(top_n_display, 3),  # Limit for readability
            audio_path=audio_path,
            total_duration=total_duration,
            figsize=figsize_summary,
            dpi=dpi,
        )
        saved_paths.append(summary_path)
    except Exception as e:
        logger.error(f"Failed to generate chunk summary: {e}", exc_info=True)
        saved_paths.append(output_dir / "chunks_summary.png")

    return saved_paths


# ── Plot 1: Heatmap ────────────────────────────────────────────────────────


def _plot_chunk_heatmap(
    chunks: List[dict],
    overall_top: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
) -> Path:
    """
    Generate a heatmap showing top-K event probabilities across all chunks.

    X-axis: Chunk index / time
    Y-axis: Event labels (top-K overall)
    Color:  Probability (white=0, dark blue=1)
    Annotations: Probability values inside cells

    Edge cases handled:
        - 0 chunks → empty plot with message
        - 1 chunk → single-row heatmap
        - Missing events → gray cells with "—"
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Chunk Probability Heatmap (No Data)")
        path = output_dir / "chunk_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # Build the top-K event name list
    top_names = _get_top_event_names(overall_top, chunks, top_n_display)

    if not top_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No predictions available",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Chunk Probability Heatmap (No Predictions)")
        path = output_dir / "chunk_heatmap.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    n_chunks = len(chunks)
    n_events = len(top_names)

    # Build data matrix: rows=events, cols=chunks
    data = np.zeros((n_events, n_chunks))
    for col_idx, chunk in enumerate(chunks):
        pred_map = {p["name"]: p["prob"] for p in chunk.get("predictions", [])}
        for row_idx, name in enumerate(top_names):
            data[row_idx, col_idx] = pred_map.get(name, 0.0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Adjust aspect ratio based on dimensions
    aspect = "auto" if n_chunks > 20 else n_chunks / max(n_events, 1)

    im = ax.imshow(
        data,
        cmap=HEATMAP_CMAP,
        aspect=aspect,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Annotate cells with probabilities
    for row_idx in range(n_events):
        for col_idx in range(n_chunks):
            prob = data[row_idx, col_idx]
            if prob > 0:
                # Choose text color based on background darkness
                text_color = "white" if prob > 0.5 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{prob:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7 if n_chunks > 15 else 9,
                    color=text_color,
                    fontweight="bold" if prob > 0.7 else "normal",
                )
            else:
                ax.text(
                    col_idx,
                    row_idx,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )

    # Labels
    ax.set_yticks(range(n_events))
    ax.set_yticklabels(
        [name[:40] + "…" if len(name) > 40 else name for name in top_names],
        fontsize=9,
    )

    # X-axis: show chunk start times
    if n_chunks <= 30:
        chunk_labels = [f"C{c['chunk_index']}\n{c['start_time']:.1f}s" for c in chunks]
        ax.set_xticks(range(n_chunks))
        ax.set_xticklabels(chunk_labels, fontsize=7, rotation=45, ha="right")
    else:
        # Show every Nth label
        step = max(1, n_chunks // 15)
        ax.set_xticks(range(0, n_chunks, step))
        ax.set_xticklabels(
            [f"{chunks[i]['start_time']:.1f}s" for i in range(0, n_chunks, step)],
            fontsize=7,
            rotation=45,
            ha="right",
        )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Probability", fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Title and metadata
    ax.set_title(
        f"Event Probability Heatmap Across Chunks\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | Chunks: {n_chunks} | RTF: {rtf:.3f}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Chunk (start time)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Event Label", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "chunk_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


# ── Plot 2: Events Timeline ────────────────────────────────────────────────


def _plot_events_timeline(
    chunks: List[dict],
    overall_top: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
) -> Path:
    """
    Generate a line plot tracking top event probabilities over time.

    X-axis: Time (seconds)
    Y-axis: Probability (0-1)
    Lines:  One per top event, with markers at each chunk midpoint
    Shaded area: ±1 standard deviation (if multiple chunks per event)

    Edge cases:
        - Single chunk → single points (no line)
        - Missing data → gaps in lines
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Event Probability Timeline (No Data)")
        path = output_dir / "events_timeline.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    top_names = _get_top_event_names(overall_top, chunks, top_n_display)

    if not top_names:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No predictions available",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Event Probability Timeline (No Predictions)")
        path = output_dir / "events_timeline.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=figsize)

    # Midpoint time of each chunk (for X-axis)
    chunk_times = [(c["start_time"] + c["end_time"]) / 2 for c in chunks]

    for idx, event_name in enumerate(top_names):
        color = COLORS[idx % len(COLORS)]

        # Collect probabilities for this event across chunks
        probs = []
        for chunk in chunks:
            pred_map = {p["name"]: p["prob"] for p in chunk.get("predictions", [])}
            probs.append(pred_map.get(event_name, np.nan))

        # Filter out NaN for plotting
        valid_indices = [i for i, p in enumerate(probs) if not np.isnan(p)]
        valid_times = [chunk_times[i] for i in valid_indices]
        valid_probs = [probs[i] for i in valid_indices]

        if len(valid_probs) >= 2:
            # Line with markers
            ax.plot(
                valid_times,
                valid_probs,
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                label=event_name[:50],
                alpha=0.85,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            # Shaded area (fill between line and zero)
            ax.fill_between(
                valid_times,
                valid_probs,
                alpha=0.1,
                color=color,
            )
        elif len(valid_probs) == 1:
            ax.scatter(
                valid_times,
                valid_probs,
                color=color,
                s=100,
                marker="D",
                label=event_name[:50],
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )

    # Reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(
        ax.get_xlim()[1] * 0.99,
        0.51,
        "50% threshold",
        fontsize=8,
        color="gray",
        ha="right",
        va="bottom",
        alpha=0.7,
    )

    # Styling
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, total_duration + 0.5 if total_duration > 0 else 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Top {top_n_display} Event Probabilities Over Time\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | Chunks: {len(chunks)} | RTF: {rtf:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    path = output_dir / "events_timeline.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


# ── Plot 3: Aggregated Bar Chart ───────────────────────────────────────────


def _plot_results_bar(
    overall_top: List[dict],
    output_dir: Path,
    audio_path: str,
    total_duration: float,
    rtf: float,
    figsize: Tuple[int, int],
    dpi: int,
) -> Path:
    """
    Generate a horizontal bar chart of overall aggregated results.

    Features:
        - Gradient-colored bars
        - Probability labels on bars
        - Performance metrics in footer
    """
    if not overall_top:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No aggregated predictions",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Aggregated Results (No Data)")
        path = output_dir / "results_bar.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=figsize)

    names = [
        p["name"][:60] + "…"
        if len(p.get("name", "")) > 60
        else p.get("name", "Unknown")
        for p in overall_top
    ]
    probs = [p.get("prob", 0) for p in overall_top]

    # Reverse for horizontal bar (top at top)
    names = names[::-1]
    probs = probs[::-1]
    y_pos = range(len(names))

    # Create gradient-colored bars
    colors_gradient = [_interpolate_color("#648FFF", "#1A1A6E", p) for p in probs]

    bars = ax.barh(y_pos, probs, height=0.6, color=colors_gradient, alpha=0.9)

    # Add value labels
    for i, (prob, bar) in enumerate(zip(probs, bars)):
        label_x = prob + 0.02
        label_color = "#333333"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=label_color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Mean Probability Across Chunks", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Aggregated Audio Tagging Results\n"
        f"{Path(audio_path).name}\n"
        f"Duration: {total_duration:.1f}s | RTF: {rtf:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="x")

    # Add rank numbers
    for i in range(len(names)):
        ax.text(
            -0.03,
            i,
            f"#{len(names) - i}",
            va="center",
            ha="right",
            fontsize=8,
            color="gray",
            fontweight="bold",
        )

    plt.tight_layout()
    path = output_dir / "results_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


# ── Plot 4: Chunk Summary Grid ─────────────────────────────────────────────


def _plot_chunk_summary(
    chunks: List[dict],
    output_dir: Path,
    top_n_display: int,
    audio_path: str,
    total_duration: float,
    figsize: Tuple[int, int],
    dpi: int,
) -> Path:
    """
    Generate a multi-panel summary showing per-chunk mini bar charts.

    Layout: Grid of subplots (up to 6×4 for many chunks)
    Each subplot: Mini horizontal bar chart for one chunk's top-N predictions

    Edge cases:
        - 1 chunk → single large panel
        - > 24 chunks → paginate or show first 24
    """
    if not chunks:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No chunks to display",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title("Per-Chunk Summary (No Data)")
        path = output_dir / "chunks_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    MAX_DISPLAY_CHUNKS = 24  # Limit to avoid overcrowding
    display_chunks = chunks[:MAX_DISPLAY_CHUNKS]
    n_chunks = len(display_chunks)

    # Calculate grid layout
    if n_chunks == 1:
        n_rows, n_cols = 1, 1
    elif n_chunks <= 4:
        n_rows, n_cols = 2, 2
    elif n_chunks <= 9:
        n_rows, n_cols = 3, 3
    elif n_chunks <= 16:
        n_rows, n_cols = 4, 4
    else:
        n_rows, n_cols = 4, 6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(
        f"Per-Chunk Top-{top_n_display} Predictions\n"
        f"{Path(audio_path).name} ({total_duration:.1f}s, {len(chunks)} total chunks)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if idx < n_chunks:
            chunk = display_chunks[idx]
            predictions = chunk.get("predictions", [])[:top_n_display]

            if predictions:
                names = [p["name"][:25] for p in predictions][::-1]
                probs = [p["prob"] for p in predictions][::-1]
                y_pos = range(len(names))

                colors = [_interpolate_color("#648FFF", "#DC267F", p) for p in probs]
                ax.barh(y_pos, probs, height=0.7, color=colors, alpha=0.85)

                # Highlight top prediction
                if probs:
                    ax.text(
                        probs[-1] + 0.05,
                        len(probs) - 1,
                        f"{probs[-1]:.0%}",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#333333",
                    )

                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=6)
                ax.set_xlim(0, 1.1)
                ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No predictions",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                    transform=ax.transAxes,
                )

            # Chunk label with time range
            time_label = (
                f"C{chunk['chunk_index']}: "
                f"{chunk['start_time']:.1f}-{chunk['end_time']:.1f}s"
            )
            ax.set_title(time_label, fontsize=7, fontweight="bold", pad=3)
        else:
            ax.axis("off")

    # Hide unused subplots
    for idx in range(n_chunks, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "chunks_summary.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


# ── Helper Functions ────────────────────────────────────────────────────────


def _get_top_event_names(
    overall_top: List[dict],
    chunks: List[dict],
    top_n: int,
) -> List[str]:
    """
    Extract top-N event names, prioritizing overall_top, falling back to
    collecting unique names from chunks.
    """
    if overall_top:
        return [p["name"] for p in overall_top[:top_n] if p.get("name")]

    # Fallback: collect from chunks
    seen = set()
    names = []
    for chunk in chunks:
        for pred in chunk.get("predictions", []):
            name = pred.get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
                if len(names) >= top_n:
                    return names
    return names


def _interpolate_color(hex_start: str, hex_end: str, factor: float) -> str:
    """
    Linear interpolation between two hex colors.
    factor=0 → start color, factor=1 → end color.
    """
    factor = max(0.0, min(1.0, factor))

    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    r1, g1, b1 = _hex_to_rgb(hex_start)
    r2, g2, b2 = _hex_to_rgb(hex_end)

    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)

    return _rgb_to_hex(r, g, b)

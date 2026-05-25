#!/usr/bin/env python3
"""
VAD Segment Scoring Module
Scores voice activity detection segments considering peak height, width, and global distribution.
"""

import json
import sys
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.traceback import install

# Install rich traceback for better error messages
install(show_locals=True)

console = Console()


class ScoringMethod(Enum):
    """Available scoring methods."""

    BALANCED = "balanced"
    WIDTH_HEAVY = "width_heavy"
    PEAK_HEAVY = "peak_heavy"
    SIMPLE = "simple"


def score_vad_segments(segment_probs: List[float], method: str = "balanced") -> float:
    """
    Score VAD segments considering peak height, width, and global distribution.

    Args:
        segment_probs: List of voice activity probabilities per segment
        method: "balanced" (default), "width_heavy", "peak_heavy", or "simple"

    Returns:
        Score between 0 and 1, higher for sustained high-confidence speech

    Examples:
        >>> score_vad_segments([0.9, 0.9, 0.9, 0.9, 0.1])  # Wide peak
        0.868
        >>> score_vad_segments([0.1, 0.1, 0.95, 0.1, 0.1])  # Narrow peak
        0.559
    """
    if not segment_probs:
        return 0.0

    probs = np.array(segment_probs)
    max_prob = float(np.max(probs))
    mean_prob = float(np.mean(probs))

    # Width score: rewards sustained high probabilities
    # Uses top 80% of probabilities to ignore brief dips
    sorted_probs = np.sort(probs)[::-1]
    top_k = max(1, int(len(sorted_probs) * 0.8))
    width_score = float(np.mean(sorted_probs[:top_k]))

    if method == "balanced":
        # Balanced: 40% width, 40% peak, 20% global
        score = 0.4 * width_score + 0.4 * max_prob + 0.2 * mean_prob

    elif method == "width_heavy":
        # Emphasizes sustained speech: 60% width, 30% peak, 10% global
        score = 0.6 * width_score + 0.3 * max_prob + 0.1 * mean_prob

    elif method == "peak_heavy":
        # Emphasizes peak confidence: 60% peak, 30% width, 10% global
        score = 0.6 * max_prob + 0.3 * width_score + 0.1 * mean_prob

    elif method == "simple":
        # Baseline: max * mean
        score = max_prob * mean_prob

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'balanced', 'width_heavy', 'peak_heavy', or 'simple'"
        )

    return float(np.clip(score, 0.0, 1.0))


def score_sustained_speech(segment_probs: List[float]) -> float:
    """Optimized for detecting sustained speech (rewards width)."""
    return score_vad_segments(segment_probs, method="width_heavy")


def score_balanced_speech(segment_probs: List[float]) -> float:
    """Balanced scoring for general voice activity."""
    return score_vad_segments(segment_probs, method="balanced")


def score_peak_confidence(segment_probs: List[float]) -> float:
    """Emphasizes peak confidence (useful for keyword spotting)."""
    return score_vad_segments(segment_probs, method="peak_heavy")


def load_probs_from_json(file_path: str) -> List[float]:
    """
    Load probabilities array from JSON file.

    Args:
        file_path: Path to JSON file containing array of floats

    Returns:
        List of probabilities

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ValueError: If data is not a list of floats
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"JSON data must be an array of floats, got {type(data).__name__}"
        )

    # Convert to float and validate range
    probs = []
    for i, val in enumerate(data):
        try:
            prob = float(val)
            if not (0.0 <= prob <= 1.0):
                console.print(
                    f"[yellow]Warning:[/yellow] Value at index {i} ({prob}) outside [0,1], clipping"
                )
                prob = np.clip(prob, 0.0, 1.0)
            probs.append(prob)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value at index {i}: {val} (must be numeric)")

    return probs


def get_score_components(segment_probs: List[float]) -> dict:
    """Get detailed scoring components for analysis."""
    if not segment_probs:
        return {}

    probs = np.array(segment_probs)
    sorted_probs = np.sort(probs)[::-1]
    top_k = max(1, int(len(sorted_probs) * 0.8))

    # Calculate various statistics
    above_06 = np.sum(probs >= 0.6)
    above_08 = np.sum(probs >= 0.8)

    return {
        "num_segments": len(probs),
        "max_probability": float(np.max(probs)),
        "min_probability": float(np.min(probs)),
        "mean_probability": float(np.mean(probs)),
        "median_probability": float(np.median(probs)),
        "std_deviation": float(np.std(probs)),
        "width_score": float(np.mean(sorted_probs[:top_k])),
        "segments_above_0.6": above_06,
        "segments_above_0.8": above_08,
        "percentile_90": float(np.percentile(probs, 90)),
        "percentile_75": float(np.percentile(probs, 75)),
        "balanced_score": score_balanced_speech(segment_probs),
        "sustained_score": score_sustained_speech(segment_probs),
        "peak_score": score_peak_confidence(segment_probs),
    }


def display_results(probs: List[float], components: dict) -> None:
    """Display scoring results with rich formatting."""

    # Header panel
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]VAD Segment Scoring Results[/bold cyan]", border_style="cyan"
        )
    )

    # Statistics table
    stats_table = Table(title="📊 Probability Statistics", title_style="bold blue")
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")
    stats_table.add_column("Interpretation", style="white")

    stats_table.add_row(
        "Number of segments", str(components["num_segments"]), "Total analysis windows"
    )
    stats_table.add_row(
        "Max probability",
        f"{components['max_probability']:.3f}",
        "Peak confidence"
        if components["max_probability"] > 0.8
        else "Moderate peak"
        if components["max_probability"] > 0.6
        else "Low peak",
    )
    stats_table.add_row(
        "Mean probability",
        f"{components['mean_probability']:.3f}",
        "Overall voice activity",
    )
    stats_table.add_row(
        "Median probability",
        f"{components['median_probability']:.3f}",
        "Typical segment confidence",
    )
    stats_table.add_row(
        "Std deviation",
        f"{components['std_deviation']:.3f}",
        "High = variable" if components["std_deviation"] > 0.3 else "Low = consistent",
    )
    stats_table.add_row(
        "Width score", f"{components['width_score']:.3f}", "Sustained activity measure"
    )

    console.print(stats_table)

    # Segments above thresholds
    threshold_table = Table(title="🎯 Threshold Analysis", title_style="bold blue")
    threshold_table.add_column("Threshold", style="cyan", no_wrap=True)
    threshold_table.add_column("Segments Above", style="green")
    threshold_table.add_column("Percentage", style="yellow")

    total = components["num_segments"]
    above_06_pct = (components["segments_above_0.6"] / total) * 100
    above_08_pct = (components["segments_above_0.8"] / total) * 100

    threshold_table.add_row(
        "≥ 0.6 (moderate confidence)",
        str(components["segments_above_0.6"]),
        f"{above_06_pct:.1f}%",
    )
    threshold_table.add_row(
        "≥ 0.8 (high confidence)",
        str(components["segments_above_0.8"]),
        f"{above_08_pct:.1f}%",
    )

    console.print(threshold_table)

    # Scoring results
    score_table = Table(title="🎯 Scoring Results", title_style="bold blue")
    score_table.add_column("Method", style="cyan", no_wrap=True)
    score_table.add_column("Score", style="green", justify="right")
    score_table.add_column("Rating", style="white")
    score_table.add_column("Best For", style="dim")

    def get_rating(score: float) -> str:
        if score >= 0.85:
            return "🟢 Excellent"
        elif score >= 0.70:
            return "🔵 Good"
        elif score >= 0.55:
            return "🟡 Marginal"
        elif score >= 0.40:
            return "🟠 Poor"
        else:
            return "🔴 Invalid"

    score_table.add_row(
        "Balanced",
        f"{components['balanced_score']:.3f}",
        get_rating(components["balanced_score"]),
        "General VAD",
    )
    score_table.add_row(
        "Width-heavy (Sustained)",
        f"{components['sustained_score']:.3f}",
        get_rating(components["sustained_score"]),
        "Sustained speech",
    )
    score_table.add_row(
        "Peak-heavy",
        f"{components['peak_score']:.3f}",
        get_rating(components["peak_score"]),
        "Keyword spotting",
    )

    console.print(score_table)

    # Visualization of probabilities
    console.print("\n[bold blue]📈 Probability Distribution[/bold blue]")

    # Create simple bar chart
    max_bar_width = 50
    probs_array = np.array(probs)

    for i, prob in enumerate(probs_array[:20]):  # Show first 20 segments
        bar_length = int(prob * max_bar_width)
        bar = "█" * bar_length
        color = "green" if prob >= 0.8 else "yellow" if prob >= 0.6 else "red"
        console.print(
            f"  [{color}]{bar:<{max_bar_width}}[/{color}] {prob:.3f} (seg {i})"
        )

    if len(probs) > 20:
        console.print(f"  [dim]... and {len(probs) - 20} more segments[/dim]")

    # Summary panel
    highest_score = max(
        components["balanced_score"],
        components["sustained_score"],
        components["peak_score"],
    )

    if highest_score >= 0.85:
        summary_color = "green"
        summary_icon = "✅"
        summary_text = "High-quality voice activity detected"
    elif highest_score >= 0.70:
        summary_color = "blue"
        summary_icon = "📢"
        summary_text = "Good voice activity detected"
    elif highest_score >= 0.55:
        summary_color = "yellow"
        summary_icon = "⚠️"
        summary_text = "Marginal voice activity - may need review"
    else:
        summary_color = "red"
        summary_icon = "❌"
        summary_text = "No significant voice activity detected"

    console.print()
    console.print(
        Panel(
            f"[bold {summary_color}]{summary_icon} {summary_text}[/bold {summary_color}]\n"
            f"Best score: [bold]{highest_score:.3f}[/bold] using {
                max(
                    [
                        (components['balanced_score'], 'balanced'),
                        (components['sustained_score'], 'sustained'),
                        (components['peak_score'], 'peak'),
                    ],
                    key=lambda x: x[0],
                )[1]
            } method",
            border_style=summary_color,
        )
    )


def save_vad_score(
    segment_probs: list[float],
    seg_dir: Path,
    seg_number: int,
) -> Path:
    """
    Save detailed VAD scoring components to a JSON file.

    Args:
        segment_probs: List of VAD probability scores per frame
        seg_dir: Directory to save the score file
        seg_number: Segment number for logging

    Returns:
        Path to the saved vad_score.json file
    """
    # Get detailed scoring components
    score_components = get_score_components(segment_probs)

    # Add segment-specific metadata
    vad_score_data = {
        "segment_number": seg_number,
        "num_frames": len(segment_probs),
        "scoring_components": score_components,
        "interpretation": _interpret_vad_score(score_components),
    }

    # Save to JSON
    vad_score_path = seg_dir / "vad_score.json"
    vad_score_path.write_text(
        json.dumps(vad_score_data, indent=2, default=float), encoding="utf-8"
    )

    return vad_score_path


def _interpret_vad_score(components: dict) -> dict:
    """
    Provide human-readable interpretation of VAD scoring components.

    Args:
        components: Dictionary of scoring components from get_score_components

    Returns:
        Dictionary with quality assessment and recommendations
    """
    balanced_score = components.get("balanced_score", 0.0)
    sustained_score = components.get("sustained_score", 0.0)
    peak_score = components.get("peak_score", 0.0)

    # Quality assessment
    if balanced_score >= 0.85:
        quality = "excellent"
        confidence = "high"
    elif balanced_score >= 0.70:
        quality = "good"
        confidence = "moderate"
    elif balanced_score >= 0.55:
        quality = "marginal"
        confidence = "low"
    elif balanced_score >= 0.40:
        quality = "poor"
        confidence = "very_low"
    else:
        quality = "invalid"
        confidence = "none"

    # Determine best use case based on scores
    scores = {
        "sustained_speech": sustained_score,
        "keyword_spotting": peak_score,
        "general_vad": balanced_score,
    }
    best_use_case = max(scores, key=scores.get)

    # Speech characteristics
    is_sustained = sustained_score > balanced_score
    has_peaks = peak_score > balanced_score

    interpretation = {
        "quality": quality,
        "confidence_level": confidence,
        "best_use_case": best_use_case.replace("_", " "),
        "speech_characteristics": {
            "is_sustained_speech": is_sustained,
            "has_clear_peaks": has_peaks,
            "consistency": "high"
            if components.get("std_deviation", 1.0) < 0.3
            else "variable",
        },
        "recommendations": _get_recommendations(components, quality),
    }

    return interpretation


def _get_recommendations(components: dict, quality: str) -> list[str]:
    """
    Generate actionable recommendations based on VAD scoring.

    Args:
        components: Scoring components dictionary
        quality: Overall quality assessment

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if quality == "excellent":
        recommendations.append("Segment contains clear, sustained speech")
        recommendations.append("Suitable for transcription or speaker recognition")
    elif quality == "good":
        recommendations.append("Speech is clearly present but may have brief gaps")
        recommendations.append("Review segment boundaries for potential trimming")
    elif quality == "marginal":
        recommendations.append("Speech presence is uncertain - may contain noise")
        recommendations.append("Consider adjusting VAD thresholds")
        recommendations.append("Manual review recommended for critical applications")
    elif quality == "poor":
        recommendations.append("Limited voice activity detected")
        recommendations.append("May be noise, music, or very quiet speech")
        recommendations.append("Consider discarding or lowering confidence thresholds")
    else:
        recommendations.append("No significant voice activity detected")
        recommendations.append("Segment likely contains silence or non-speech audio")

    # Add specific recommendations based on components
    mean_prob = components.get("mean_probability", 0.0)
    std_dev = components.get("std_deviation", 0.0)

    if mean_prob < 0.3:
        recommendations.append(
            "Very low mean probability suggests minimal speech content"
        )
    if std_dev > 0.3:
        recommendations.append(
            "High variability indicates inconsistent speech presence"
        )

    if (
        components.get("segments_above_0.8", 0)
        > components.get("num_segments", 1) * 0.5
    ):
        recommendations.append(
            "Strong high-confidence regions present - focus analysis here"
        )

    return recommendations


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Score VAD segments from a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s probs.json
  %(prog)s probs.json --method width_heavy
  %(prog)s probs.json --components
  %(prog)s --test
        """,
    )

    parser.add_argument(
        "probs_path",
        type=str,
        nargs="?",
        help="Path to JSON file containing array of probabilities",
    )

    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="balanced",
        choices=["balanced", "width_heavy", "peak_heavy", "simple"],
        help="Scoring method (default: balanced)",
    )

    parser.add_argument(
        "--components",
        "-c",
        action="store_true",
        help="Show detailed component analysis",
    )

    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress rich output, print only the score",
    )

    parser.add_argument(
        "--test", action="store_true", help="Run with built-in test cases"
    )

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        console.print("[bold cyan]Running built-in tests...[/bold cyan]\n")

        test_cases = {
            "narrow_high": [0.1, 0.1, 0.95, 0.1, 0.1],
            "wide_high": [0.9, 0.9, 0.9, 0.9, 0.1],
            "wide_moderate": [0.6, 0.6, 0.6, 0.6, 0.2],
        }

        results = []
        for name, probs in test_cases.items():
            components = get_score_components(probs)
            results.append({"name": name, "probs": probs, "components": components})

            if not args.quiet:
                console.print(f"[bold]{name.upper()}[/bold]: {probs}")
                console.print(f"  Balanced: {components['balanced_score']:.3f}")
                console.print(f"  Sustained: {components['sustained_score']:.3f}")
                console.print(f"  Peak-heavy: {components['peak_score']:.3f}\n")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]✓ Results saved to {args.output}[/green]")

        return

    # Check if probs_path is provided
    if not args.probs_path:
        parser.print_help()
        console.print(
            "\n[red]Error: Either provide a JSON file path or use --test[/red]"
        )
        sys.exit(1)

    # Load probabilities
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading probabilities...", total=None)
            probs = load_probs_from_json(args.probs_path)
            progress.update(task, completed=True)

        console.print(
            f"[green]✓ Loaded {len(probs)} probabilities from {args.probs_path}[/green]"
        )

        # Calculate score or components
        if args.components:
            components = get_score_components(probs)
            if args.quiet:
                # Just print the score for the requested method
                score = score_vad_segments(probs, method=args.method)
                print(f"{score:.6f}")
            else:
                display_results(probs, components)
        else:
            # Just calculate the requested method
            score = score_vad_segments(probs, method=args.method)

            if args.quiet:
                print(f"{score:.6f}")
            else:
                console.print(
                    f"\n[bold cyan]VAD Score ({args.method}):[/bold cyan] [bold green]{score:.3f}[/bold green]"
                )

                # Show quick interpretation
                if score >= 0.85:
                    console.print(
                        "[green]✅ Excellent - Sustained high-confidence speech[/green]"
                    )
                elif score >= 0.70:
                    console.print("[blue]📢 Good - Clear voice activity[/blue]")
                elif score >= 0.55:
                    console.print(
                        "[yellow]⚠️ Marginal - Possible voice or noisy speech[/yellow]"
                    )
                elif score >= 0.40:
                    console.print(
                        "[orange1]❌ Poor - Unlikely voice activity[/orange1]"
                    )
                else:
                    console.print("[red]🔴 Invalid - No voice activity detected[/red]")

        # Save results if output specified
        if args.output:
            result = {
                "file": args.probs_path,
                "method": args.method,
                "score": score_vad_segments(probs, method=args.method),
                "num_segments": len(probs),
                "statistics": {
                    "max": float(np.max(probs)),
                    "mean": float(np.mean(probs)),
                    "min": float(np.min(probs)),
                    "std": float(np.std(probs)),
                },
            }

            if args.components:
                result["components"] = components

            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

            console.print(f"\n[green]✓ Results saved to {args.output}[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()

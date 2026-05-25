#!/usr/bin/env python3
"""
segment_scoring.py
==================
Reusable scoring utilities for List[float] segment probability sequences.

Metrics
-------
- peak_height       : max probability value
- peak_width        : longest run of segments above a threshold
- peak_area         : sum of probabilities inside the widest peak
- peak_prominence   : peak_height minus the surrounding baseline mean
- global_mean       : average of all probabilities
- global_median     : median of all probabilities
- global_std        : standard deviation
- coverage          : fraction of segments above threshold
- composite_score   : weighted combination of the above
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.traceback import install

# Install rich traceback for better error messages
install(show_locals=True)

console = Console()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Peak:
    start: int  # inclusive index
    end: int  # inclusive index
    height: float  # max value in this peak
    width: int  # number of segments in this peak
    area: float  # sum of probabilities in this peak
    prominence: float  # height minus surrounding baseline mean


@dataclass
class SegmentScore:
    # Peak metrics
    peak_height: float
    peak_width: int
    peak_area: float
    peak_prominence: float

    # Global metrics
    global_mean: float
    global_median: float
    global_std: float
    coverage: float  # fraction of segments above threshold

    # Final score
    composite: float

    # Detail
    threshold: float
    peaks: List[Peak] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["peaks"] = [asdict(p) for p in self.peaks]
        return d


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _find_peaks(probs: List[float], threshold: float) -> List[Peak]:
    """Return all contiguous runs above *threshold* as Peak objects."""
    peaks: List[Peak] = []
    n = len(probs)
    i = 0
    while i < n:
        if probs[i] >= threshold:
            start = i
            while i < n and probs[i] >= threshold:
                i += 1
            end = i - 1
            segment = probs[start : end + 1]
            height = max(segment)
            width = end - start + 1
            area = sum(segment)

            # Baseline: values outside [start, end]
            outside = probs[:start] + probs[end + 1 :]
            baseline = statistics.mean(outside) if outside else 0.0
            prominence = height - baseline

            peaks.append(Peak(start, end, height, width, area, prominence))
        else:
            i += 1
    return peaks


def _auto_threshold(probs: List[float]) -> float:
    """Default threshold: mean + 0.25 * std, clamped to [0.1, 0.9]."""
    if not probs:
        return 0.5
    mu = statistics.mean(probs)
    sigma = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    return max(0.1, min(0.9, mu + 0.25 * sigma))


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def score_segment_probs(
    probs: List[float],
    *,
    threshold: Optional[float] = None,
    weights: Optional[dict] = None,
) -> SegmentScore:
    """
    Compute a composite score for a segment probability list.

    Parameters
    ----------
    probs       : List of floats in [0, 1].
    threshold   : Minimum probability to count as part of a peak.
                  Defaults to mean + 0.25*std (auto).
    weights     : Dict with keys from
                  {'peak_area', 'peak_prominence', 'global_mean',
                   'global_median', 'coverage', 'stability'}
                  and float values (will be normalised to sum=1).

    Returns
    -------
    SegmentScore dataclass.
    """
    if not probs:
        raise ValueError("probs must be a non-empty list")
    if not all(0.0 <= p <= 1.0 for p in probs):
        raise ValueError("All probabilities must be in [0, 1]")

    # ---- threshold --------------------------------------------------------
    t = threshold if threshold is not None else _auto_threshold(probs)

    # ---- global stats -----------------------------------------------------
    mu = statistics.mean(probs)
    median = statistics.median(probs)
    sigma = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    coverage = sum(1 for p in probs if p >= t) / len(probs)

    # ---- peaks ------------------------------------------------------------
    peaks = _find_peaks(probs, t)

    if peaks:
        best = max(peaks, key=lambda pk: pk.area)
        peak_height = best.height
        peak_width = best.width
        peak_area = best.area
        peak_prominence = best.prominence
    else:
        peak_height = max(probs)
        peak_width = 0
        peak_area = 0.0
        peak_prominence = peak_height - mu

    # ---- normalised sub-scores (all in [0,1]) ----------------------------
    # peak_area_norm: normalise by maximum possible area (all 1.0)
    max_possible_area = len(probs) * 1.0
    peak_area_norm = peak_area / max_possible_area if max_possible_area > 0 else 0.0
    peak_prominence_norm = max(0.0, min(1.0, peak_prominence))
    stability = max(0.0, 1.0 - sigma)  # low variance = stable

    # ---- weights ----------------------------------------------------------
    default_weights = {
        "peak_area": 0.30,
        "peak_prominence": 0.25,
        "global_mean": 0.20,
        "global_median": 0.10,
        "coverage": 0.10,
        "stability": 0.05,
    }
    w = {**default_weights, **(weights or {})}
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}  # normalise

    composite = (
        w["peak_area"] * peak_area_norm
        + w["peak_prominence"] * peak_prominence_norm
        + w["global_mean"] * mu
        + w["global_median"] * median
        + w["coverage"] * coverage
        + w["stability"] * stability
    )

    return SegmentScore(
        peak_height=round(peak_height, 4),
        peak_width=peak_width,
        peak_area=round(peak_area, 4),
        peak_prominence=round(peak_prominence, 4),
        global_mean=round(mu, 4),
        global_median=round(median, 4),
        global_std=round(sigma, 4),
        coverage=round(coverage, 4),
        composite=round(composite, 4),
        threshold=round(t, 4),
        peaks=peaks,
    )


# ---------------------------------------------------------------------------
# Convenience: rank multiple candidates
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: List[List[float]],
    *,
    threshold: Optional[float] = None,
    weights: Optional[dict] = None,
    labels: Optional[List[str]] = None,
) -> List[Tuple[str, float, SegmentScore]]:
    """
    Score and rank a list of segment_probs candidates.

    Returns a list of (label, composite_score, SegmentScore) tuples,
    sorted best-first.
    """
    if labels is None:
        labels = [f"candidate_{i}" for i in range(len(candidates))]
    if len(labels) != len(candidates):
        raise ValueError("labels length must match candidates length")

    results = []
    for label, probs in zip(labels, candidates):
        s = score_segment_probs(probs, threshold=threshold, weights=weights)
        results.append((label, s.composite, s))

    return sorted(results, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Load from JSON
# ---------------------------------------------------------------------------


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
                prob = max(0.0, min(1.0, prob))
            probs.append(prob)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value at index {i}: {val} (must be numeric)")

    return probs


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------


def display_score(score: SegmentScore, label: str = "Result") -> None:
    """Display scoring results with rich formatting."""

    # Header panel
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Segment Scoring Results: {label}[/bold cyan]",
            border_style="cyan",
        )
    )

    # Peak metrics table
    peak_table = Table(title="📈 Peak Metrics", title_style="bold blue")
    peak_table.add_column("Metric", style="cyan", no_wrap=True)
    peak_table.add_column("Value", style="green")
    peak_table.add_column("Interpretation", style="white")

    peak_table.add_row(
        "Threshold", f"{score.threshold:.3f}", "Minimum probability for peak detection"
    )
    peak_table.add_row(
        "Peak height", f"{score.peak_height:.3f}", "Max confidence in best peak"
    )
    peak_table.add_row(
        "Peak width", f"{score.peak_width} segments", "Duration of best peak"
    )
    peak_table.add_row(
        "Peak area", f"{score.peak_area:.3f}", "Total confidence mass in best peak"
    )
    peak_table.add_row(
        "Peak prominence",
        f"{score.peak_prominence:.3f}",
        "Height above surrounding baseline",
    )

    console.print(peak_table)

    # Global metrics table
    global_table = Table(title="🌍 Global Metrics", title_style="bold blue")
    global_table.add_column("Metric", style="cyan", no_wrap=True)
    global_table.add_column("Value", style="green")
    global_table.add_column("Interpretation", style="white")

    global_table.add_row("Mean", f"{score.global_mean:.3f}", "Overall confidence level")
    global_table.add_row(
        "Median", f"{score.global_median:.3f}", "Typical segment confidence"
    )
    global_table.add_row(
        "Std deviation", f"{score.global_std:.3f}", "High = variable, low = consistent"
    )
    global_table.add_row(
        "Coverage",
        f"{score.coverage:.1%}",
        f"Segments above {score.threshold:.2f} threshold",
    )

    console.print(global_table)

    # Final score
    score_color = (
        "green"
        if score.composite >= 0.7
        else "yellow"
        if score.composite >= 0.5
        else "red"
    )
    console.print()
    console.print(
        Panel(
            f"[bold {score_color}]Composite Score: {score.composite:.3f}[/bold {score_color}]",
            border_style=score_color,
            title="🎯 Final Score",
        )
    )

    # Peaks found
    if score.peaks:
        peak_list = Table(
            title=f"🔍 Peaks Found ({len(score.peaks)})", title_style="bold blue"
        )
        peak_list.add_column("Peak", style="cyan", no_wrap=True)
        peak_list.add_column("Range", style="green")
        peak_list.add_column("Height", style="yellow", justify="right")
        peak_list.add_column("Width", style="yellow", justify="right")
        peak_list.add_column("Area", style="yellow", justify="right")
        peak_list.add_column("Prominence", style="yellow", justify="right")

        for i, pk in enumerate(score.peaks):
            peak_list.add_row(
                f"#{i + 1}",
                f"[{pk.start}–{pk.end}]",
                f"{pk.height:.3f}",
                str(pk.width),
                f"{pk.area:.3f}",
                f"{pk.prominence:.3f}",
            )

        console.print(peak_list)

    # Interpretation
    console.print()
    if score.composite >= 0.85:
        console.print(
            "[bold green]✅ Excellent signal quality - Strong, sustained activity detected[/bold green]"
        )
    elif score.composite >= 0.70:
        console.print(
            "[bold blue]📢 Good signal quality - Clear activity with good confidence[/bold blue]"
        )
    elif score.composite >= 0.55:
        console.print(
            "[bold yellow]⚠️ Moderate signal quality - Some activity detected, may need review[/bold yellow]"
        )
    elif score.composite >= 0.40:
        console.print(
            "[bold orange1]❌ Poor signal quality - Weak or intermittent activity[/bold orange1]"
        )
    else:
        console.print(
            "[bold red]🔴 Invalid signal - No significant activity detected[/bold red]"
        )


def print_score(score: SegmentScore, label: str = "Result") -> None:
    """Legacy simple text printer for backward compatibility."""
    sep = "─" * 44
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Threshold          : {score.threshold}")
    print("  ── Peak metrics ──────────────────────")
    print(f"  Peak height        : {score.peak_height}")
    print(f"  Peak width         : {score.peak_width} segments")
    print(f"  Peak area          : {score.peak_area}")
    print(f"  Peak prominence    : {score.peak_prominence}")
    print("  ── Global metrics ────────────────────")
    print(f"  Mean               : {score.global_mean}")
    print(f"  Median             : {score.global_median}")
    print(f"  Std dev            : {score.global_std}")
    print(f"  Coverage           : {score.coverage:.1%}")
    print("  ── Final ─────────────────────────────")
    print(f"  Composite score    : {score.composite}")
    if score.peaks:
        print(f"  Peaks found        : {len(score.peaks)}")
        for i, pk in enumerate(score.peaks):
            print(
                f"    [{i}] idx {pk.start}–{pk.end}  "
                f"h={pk.height:.3f}  w={pk.width}  "
                f"area={pk.area:.3f}  prom={pk.prominence:.3f}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Score segment probabilities from a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s probs.json
  %(prog)s probs.json --threshold 0.6
  %(prog)s probs.json --weights '{"peak_area": 0.5, "global_mean": 0.5}'
  %(prog)s --test
  %(prog)s --rank candidates.json
        """,
    )

    parser.add_argument(
        "probs_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to JSON file containing array of probabilities",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Threshold for peak detection (default: auto = mean + 0.25*std)",
    )

    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="JSON string of weights for scoring components",
    )

    parser.add_argument(
        "--test", action="store_true", default=False, help="Run built-in test cases"
    )

    parser.add_argument(
        "--rank",
        "-r",
        type=str,
        default=None,
        help="JSON file with multiple candidate arrays for ranking",
    )

    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save results to JSON file"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress rich output, print only the composite score",
    )

    parser.add_argument(
        "--simple",
        "-s",
        action="store_true",
        default=False,
        help="Use simple text output instead of rich formatting",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show verbose output with all details",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if not args.test and not args.rank and not args.probs_path:
        parser.print_help()
        console.print("\n[red]Error: Provide a JSON file path, --test, or --rank[/red]")
        sys.exit(1)

    if args.probs_path and args.rank:
        console.print("[red]Error: Cannot specify both probs_path and --rank[/red]")
        sys.exit(1)

    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        console.print("[red]Error: Threshold must be between 0.0 and 1.0[/red]")
        sys.exit(1)

    # Parse weights if provided
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
            # Validate weight keys
            valid_keys = {
                "peak_area",
                "peak_prominence",
                "global_mean",
                "global_median",
                "coverage",
                "stability",
            }
            for key in weights.keys():
                if key not in valid_keys:
                    console.print(
                        f"[yellow]Warning: Unknown weight key '{key}'[/yellow]"
                    )
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing weights JSON: {e}[/red]")
            sys.exit(1)

    # Run tests if requested
    if args.test:
        console.print("[bold cyan]Running built-in tests...[/bold cyan]\n")

        test_cases = {
            "well_shaped_signal": [
                0.08,
                0.12,
                0.18,
                0.55,
                0.72,
                0.88,
                0.90,
                0.85,
                0.78,
                0.60,
                0.30,
                0.15,
                0.10,
                0.07,
            ],
            "narrow_spike": [0.1] * 10 + [0.95] + [0.1] * 10,
            "wide_plateau": [0.1] * 3 + [0.70] * 15 + [0.1] * 3,
            "noisy": [0.3, 0.9, 0.2, 0.85, 0.1, 0.88, 0.15, 0.7],
            "consistent": [0.65, 0.70, 0.68, 0.72, 0.69, 0.71, 0.67, 0.70],
            "weak": [0.2, 0.25, 0.22, 0.30, 0.28, 0.24, 0.21, 0.23],
        }

        results = []
        for name, probs in test_cases.items():
            score = score_segment_probs(
                probs, threshold=args.threshold, weights=weights
            )
            results.append(
                {
                    "name": name,
                    "composite": score.composite,
                    "peak_height": score.peak_height,
                    "peak_width": score.peak_width,
                    "coverage": score.coverage,
                    "score": score.as_dict(),
                }
            )

            if args.quiet:
                print(f"{name}: {score.composite:.4f}")
            elif args.simple:
                print_score(score, name.replace("_", " ").title())
            else:
                display_score(score, name.replace("_", " ").title())

        # Print summary table for test mode
        if not args.quiet and not args.simple:
            summary_table = Table(title="📊 Test Summary", title_style="bold green")
            summary_table.add_column("Case", style="cyan")
            summary_table.add_column("Composite", style="yellow", justify="right")
            summary_table.add_column("Peak Height", style="green", justify="right")
            summary_table.add_column("Peak Width", style="blue", justify="right")
            summary_table.add_column("Coverage", style="white", justify="right")

            for result in results:
                summary_table.add_row(
                    result["name"].replace("_", " ").title(),
                    f"{result['composite']:.3f}",
                    f"{result['peak_height']:.3f}",
                    str(result["peak_width"]),
                    f"{result['coverage']:.1%}",
                )

            console.print("\n")
            console.print(summary_table)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]✓ Results saved to {args.output}[/green]")

        return

    # Handle ranking mode
    if args.rank:
        try:
            with open(args.rank, "r") as f:
                candidates_data = json.load(f)

            if isinstance(candidates_data, dict):
                labels = list(candidates_data.keys())
                candidates = list(candidates_data.values())
            elif isinstance(candidates_data, list):
                labels = None
                candidates = candidates_data
            else:
                raise ValueError("Rank file must contain object or array")

            if not candidates:
                raise ValueError("No candidates found in rank file")

            ranking = rank_candidates(
                candidates, threshold=args.threshold, weights=weights, labels=labels
            )

            if args.quiet:
                for label, score, _ in ranking:
                    print(f"{label}: {score:.4f}")
            elif args.simple:
                print("\n" + "═" * 60)
                print("  CANDIDATE RANKING")
                print("═" * 60)
                for rank, (label, score, score_obj) in enumerate(ranking, 1):
                    print(f"\n  #{rank}: {label}")
                    print(f"    Composite: {score:.4f}")
                    print(f"    Peak height: {score_obj.peak_height:.3f}")
                    print(f"    Peak width: {score_obj.peak_width}")
                    print(f"    Coverage: {score_obj.coverage:.1%}")
                print("\n" + "═" * 60)
            else:
                rank_table = Table(
                    title="🏆 Candidate Ranking", title_style="bold blue"
                )
                rank_table.add_column("Rank", style="cyan", no_wrap=True)
                rank_table.add_column("Label", style="green")
                rank_table.add_column("Composite", style="yellow", justify="right")
                rank_table.add_column("Peak Height", style="white", justify="right")
                rank_table.add_column("Peak Width", style="white", justify="right")
                rank_table.add_column("Coverage", style="white", justify="right")

                for rank, (label, score, score_obj) in enumerate(ranking, 1):
                    rank_table.add_row(
                        f"#{rank}",
                        label,
                        f"{score:.4f}",
                        f"{score_obj.peak_height:.3f}",
                        str(score_obj.peak_width),
                        f"{score_obj.coverage:.1%}",
                    )

                console.print(rank_table)

                # Show best candidate recommendation
                if ranking:
                    best_label, best_score, best_obj = ranking[0]
                    console.print()
                    if best_score >= 0.7:
                        console.print(
                            f"[bold green]✅ Recommended: '{best_label}' with score {best_score:.3f}[/bold green]"
                        )
                    elif best_score >= 0.5:
                        console.print(
                            f"[bold yellow]⚠️ Best candidate: '{best_label}' with score {best_score:.3f} (needs review)[/bold yellow]"
                        )
                    else:
                        console.print(
                            f"[bold red]❌ Best candidate: '{best_label}' with score {best_score:.3f} (poor quality)[/bold red]"
                        )

            if args.output:
                output_data = [
                    {
                        "rank": i,
                        "label": label,
                        "composite": score,
                        "peak_height": score_obj.peak_height,
                        "peak_width": score_obj.peak_width,
                        "coverage": score_obj.coverage,
                        "details": score_obj.as_dict(),
                    }
                    for i, (label, score, score_obj) in enumerate(ranking, 1)
                ]
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                if not args.quiet:
                    console.print(f"\n[green]✓ Ranking saved to {args.output}[/green]")

        except FileNotFoundError as e:
            console.print(f"[red]Error: File not found - {e}[/red]")
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing JSON: {e}[/red]")
            sys.exit(1)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            if args.verbose:
                console.print_exception()
            sys.exit(1)

        return

    # Single file mode
    if not args.probs_path:
        parser.print_help()
        console.print("\n[red]Error: Provide a JSON file path, --test, or --rank[/red]")
        sys.exit(1)

    # Load and score
    try:
        # Check if file exists
        if not Path(args.probs_path).exists():
            raise FileNotFoundError(f"File not found: {args.probs_path}")

        # Load with progress indicator for non-quiet mode
        if args.quiet:
            probs = load_probs_from_json(args.probs_path)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Loading probabilities...", total=None)
                probs = load_probs_from_json(args.probs_path)
                progress.update(task, completed=True)

        if not args.quiet:
            console.print(
                f"[green]✓ Loaded {len(probs)} probabilities from {args.probs_path}[/green]"
            )

            # Show basic stats
            probs_array = np.array(probs)
            console.print(
                f"  Range: [{np.min(probs_array):.3f}, {np.max(probs_array):.3f}]"
            )
            console.print(
                f"  Mean: {np.mean(probs_array):.3f} ± {np.std(probs_array):.3f}"
            )

        # Calculate score
        score = score_segment_probs(probs, threshold=args.threshold, weights=weights)

        # Output results
        if args.quiet:
            print(f"{score.composite:.6f}")
        elif args.simple:
            print_score(score, Path(args.probs_path).stem.replace("_", " ").title())
        else:
            display_score(score, Path(args.probs_path).stem.replace("_", " ").title())

        # Save output if requested
        if args.output:
            output_data = {
                "file": args.probs_path,
                "num_segments": len(probs),
                "threshold": args.threshold
                if args.threshold is not None
                else score.threshold,
                "weights": weights,
                "composite": score.composite,
                "score": score.as_dict(),
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            if not args.quiet:
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
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()

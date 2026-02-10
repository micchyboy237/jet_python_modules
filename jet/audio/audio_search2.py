# audio_search.py
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from rich.console import Console
from scipy import signal

console = Console()


def load_audio_metadata_and_resampled(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Load audio → return metadata + resampled mono float32 array
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio = AudioSegment.from_file(str(path))
        original_sr = audio.frame_rate

        if target_sr is None:
            # sensible auto choice
            target_sr = original_sr
            if target_sr > 32000:
                target_sr = 22050
            console.print(f"[dim]Auto selected working sample rate: {target_sr} Hz[/]")
        else:
            console.print(f"[dim]Using requested sample rate: {target_sr} Hz[/]")

        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 32768.0
        return samples, target_sr, original_sr
    except CouldntDecodeError as e:
        raise CouldntDecodeError(f"Cannot decode {path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load {path}: {e}") from e


def normalize_energy(x: np.ndarray) -> np.ndarray:
    """Remove DC offset & normalize energy (unit variance)"""
    x = x - np.mean(x)
    energy = np.sqrt(np.mean(x**2))
    if energy > 1e-10:
        x /= energy
    return x


def find_audio_clip(
    haystack_path: str | Path,
    needle_path: str | Path,
    working_sr: int | None = None,
    min_similarity: float = 0.75,
    top_n: int = 3,
) -> list[tuple[float, float]]:
    """
    Search for needle audio inside haystack.
    """
    haystack, sr, haystack_orig_sr = load_audio_metadata_and_resampled(
        haystack_path, working_sr
    )
    needle, _, needle_orig_sr = load_audio_metadata_and_resampled(
        needle_path, working_sr
    )

    if len(needle) >= len(haystack):
        console.print(
            "[yellow]Warning: needle is longer or equal to haystack → swapping[/]"
        )
        haystack, needle = needle, haystack

    if len(needle) < 10:
        raise ValueError("Needle is too short (<10 samples)")

    # ── rest of function unchanged ──
    haystack = normalize_energy(haystack)
    needle = normalize_energy(needle)

    corr = signal.correlate(haystack, needle, mode="valid", method="fft")
    corr /= np.max(np.abs(corr)) + 1e-12

    peaks_idx = np.argsort(corr)[::-1]

    results = []
    needle_len_sec = len(needle) / sr

    for idx in peaks_idx[: top_n * 2]:
        sim = corr[idx]
        if sim < min_similarity:
            break
        start_sec = idx / sr
        results.append((start_sec, float(sim)))
        if len(results) >= top_n:
            break

    return results[:top_n]


def seconds_to_hms(seconds: float) -> str:
    """0.0 → '0:00'    65.4 → '1:05'    3661 → '1:01:01'"""
    if seconds < 0:
        return "0:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ────────────────────────────────────────────────
#                   CLI
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find short audio clip inside longer file"
    )
    parser.add_argument("haystack", type=str, help="Long reference audio file")
    parser.add_argument("needle", type=str, help="Short clip to search for")
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Working / processing sample rate (default: auto ≈22050)",
    )
    parser.add_argument("--thresh", type=float, default=0.75, help="Min similarity 0-1")
    parser.add_argument("--top", type=int, default=3, help="Show top N matches")

    args = parser.parse_args()

    try:
        matches = find_audio_clip(
            args.haystack,
            args.needle,
            working_sr=args.sr,
            min_similarity=args.thresh,
            top_n=args.top,
        )

        if not matches:
            console.print(
                "[red]No good matches found.[/] Try lower --thresh or check files."
            )
        else:
            console.print(f"\n[bold cyan]Top matches in {args.haystack}:[/]")
            for i, (start, sim) in enumerate(matches, 1):
                console.print(
                    f"  [green]{i}.[/] {seconds_to_hms(start)} – "
                    f"similarity [bold]{sim:.3f}[/]"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")

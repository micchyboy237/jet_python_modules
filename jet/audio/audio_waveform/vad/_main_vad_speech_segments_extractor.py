import json
from pathlib import Path

import matplotlib
from jet.audio.audio_waveform.vad.vad_utils import save_segments

matplotlib.use("Agg")
import argparse
import shutil

from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
)
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    from jet.audio.audio_waveform.vad.vad_speech_segments_extractor import (
        extract_speech_audio,
        extract_speech_timestamps,
    )

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_1_speaker.wav"
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD + hybrid pre-roll"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"speech threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-ms",
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help=f"minimum silence duration in seconds (default: {DEFAULT_MIN_SILENCE_SEC})",
    )
    parser.add_argument(
        "-mp",
        "--min-speech",
        type=float,
        default=DEFAULT_MIN_SPEECH_SEC,
        help=f"minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SEC})",
    )
    parser.add_argument(
        "-mx",
        "--max-speech",
        type=float,
        default=8.0,
        help="maximum speech duration in seconds",
    )
    parser.add_argument(
        "-sw",
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW_SIZE,
        help=f"smoothing window size (default: {DEFAULT_SMOOTH_WINDOW_SIZE})",
    )
    parser.add_argument(
        "-mb",
        "--max-buffer-sec",
        type=float,
        default=DEFAULT_MAX_BUFFER_SEC,
        help=f"stream buffer duration in seconds (default: {DEFAULT_MAX_BUFFER_SEC})",
    )
    # Pre-roll args
    parser.add_argument(
        "--preroll-max-sec",
        type=float,
        default=DEFAULT_PREROLL_MAX_SEC,
        help=f"max pre-roll look-back in seconds (default: {DEFAULT_PREROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--preroll-threshold",
        type=float,
        default=DEFAULT_PREROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for pre-roll extension (default: {DEFAULT_PREROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--preroll-prob-weight",
        type=float,
        default=DEFAULT_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--preroll-rms-weight",
        type=float,
        default=DEFAULT_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_RMS_WEIGHT})",
    )
    # Post-roll args
    parser.add_argument(
        "--postroll-max-sec",
        type=float,
        default=DEFAULT_POSTROLL_MAX_SEC,
        help=f"max post-roll look-forward in seconds (default: {DEFAULT_POSTROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--postroll-threshold",
        type=float,
        default=DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for post-roll extension (default: {DEFAULT_POSTROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--postroll-prob-weight",
        type=float,
        default=DEFAULT_PROB_WEIGHT,
        help=f"weight for speech prob in hybrid score (default: {DEFAULT_PROB_WEIGHT})",
    )
    parser.add_argument(
        "--postroll-rms-weight",
        type=float,
        default=DEFAULT_RMS_WEIGHT,
        help=f"weight for RMS energy in hybrid score (default: {DEFAULT_RMS_WEIGHT})",
    )
    parser.add_argument(
        "--soft-limit",
        type=float,
        default=DEFAULT_SOFT_LIMIT_SEC,
        help=f"Soft max segment duration before valley-trough splitting (default: {DEFAULT_SOFT_LIMIT_SEC}s; 0 = disabled)",
    )
    parser.add_argument(
        "--soft-limit-min-valley",
        type=float,
        default=DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
        help="Min silence duration for a split-candidate valley (default: %(default)ss)",
    )
    parser.add_argument(
        "--soft-limit-smoothing",
        type=int,
        default=DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
        help="Smoothing window for soft-limit trough detection (default: %(default)s frames)",
    )
    parser.add_argument(
        "--soft-limit-prominence",
        type=float,
        default=DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
        help="Min trough prominence for soft-limit splitting (default: %(default)s)",
    )
    parser.add_argument(
        "--soft-limit-offset",
        type=float,
        default=DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        help="Trough must be >= this many seconds into the segment (default: %(default)ss)",
    )

    args = parser.parse_args()
    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    console.rule("Audio Segmenter – FireRedVAD2 + Hybrid Pre/Post-Roll", style="blue")
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_path).name}\n")
    console.print(
        f"[dim]Pre-roll:  max={args.preroll_max_sec}s  "
        f"threshold={args.preroll_threshold}  "
        f"prob_w={args.preroll_prob_weight}  "
        f"rms_w={args.preroll_rms_weight}[/dim]"
    )
    console.print(
        f"[dim]Post-roll: max={args.postroll_max_sec}s  "
        f"threshold={args.postroll_threshold}  "
        f"prob_w={args.postroll_prob_weight}  "
        f"rms_w={args.postroll_rms_weight}[/dim]\n"
    )

    # ── Step 1: detect segments (with per-frame probabilities) ────────────
    segments, speech_probs = extract_speech_timestamps(
        audio_path,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        return_seconds=True,
        with_scores=True,
        include_non_speech=False,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        seg_type = seg["type"]
        type_color = "bold green" if seg_type == "speech" else "bold red"
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white]"
            f" - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"dur=[bold magenta]{seg['duration']:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"type=[{type_color}]{seg_type}[/{type_color}]"
        )

    if not any(s["type"] == "speech" for s in segments):
        console.print("[red]No speech segments found.[/red]")
        raise SystemExit(0)

    # ── Step 2: extract raw audio for each speech segment ─────────────────
    audio_chunks = extract_speech_audio(
        audio_path,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        smooth_window_size=args.smooth_window,
        max_buffer_sec=args.max_buffer_sec,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
    )

    # ── Step 3: save everything to disk ───────────────────────────────────
    saved_metas = save_segments(segments, audio_chunks, output_dir)

    # ── Step 4: write summary JSON files ──────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        slim = [
            {k: v for k, v in m.items() if k != "segment_probs"} for m in saved_metas
        ]
        json.dump(slim, fh, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]✓ Summary saved to:[/bold green] "
        f"[link=file://{summary_path.resolve()}]{summary_path}[/link]"
    )

    all_probs_path = output_dir / "speech_probs.json"
    with open(all_probs_path, "w", encoding="utf-8") as fh:
        json.dump(
            speech_probs if isinstance(speech_probs, list) else [],
            fh,
            indent=2,
        )
    console.print(
        f"[bold green]✓ Full probs saved to:[/bold green] "
        f"[link=file://{all_probs_path.resolve()}]{all_probs_path}[/link]"
    )

    console.rule("Done", style="green")

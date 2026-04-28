"""
Japanese Live Speech-to-English Translation

Uses:

  1. JapaneseSpeechTranscriber (Zipformer ReazonSpeech by default) for ASR
  2. JapaneseToEnglishTranslator (llama-cpp-python + GGUF) for translation

Usage:

python /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/sherpa_onnx/helpers/streaming_speech_waves_jp_en.py \
  --silero-vad-model /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/vad/silero_vad.onnx \
  --tokens /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/tokens.txt \
  --encoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/encoder-epoch-35-avg-1.int8.onnx \
  --decoder /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/decoder-epoch-35-avg-1.int8.onnx \
  --joiner /Users/jethroestrada/.cache/pretrained_models/sherpa-onnx/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/joiner-epoch-35-avg-1.int8.onnx \
  --num-threads 4 \
  --sample-rate 16000 \
  --feature-dim 80

"""

from __future__ import annotations

import argparse
import queue
import shutil
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.speech.firered.speech_waves_tracker import SpeechWavesTracker
from jet.libs.sherpa_onnx.helpers.ja_en_translator_http import translate
from jet.libs.sherpa_onnx.helpers.ja_transcriber import JapaneseSpeechTranscriber
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------


def _sec_to_srt_timestamp(seconds: float) -> str:
    """Convert a float number of seconds to SRT timestamp HH:MM:SS,mmm."""
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def save_srt(
    wave_dir: Path,
    start_sec: float,
    end_sec: float,
    japanese_text: str,
    english_text: str,
) -> Path:
    """
    Write a two-entry SRT file into wave_dir/transcript.srt.

    Entry 1 = Japanese transcription (original timestamps).
    Entry 2 = English translation   (same timestamps, index 2).
    """
    srt_path = wave_dir / "transcript.srt"
    start_ts = _sec_to_srt_timestamp(start_sec)
    end_ts = _sec_to_srt_timestamp(end_sec)
    lines = [
        "1",
        f"{start_ts} --> {end_ts}",
        japanese_text,
        "",
        "2",
        f"{start_ts} --> {end_ts}",
        english_text,
        "",
    ]
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


# ---------------------------------------------------------------------------
# Extended argument parser (adds ASR / translator args on top of tracker args)
# ---------------------------------------------------------------------------


def get_full_args(default_output_dir: Path) -> argparse.Namespace:
    """
    Combine tracker CLI args with the ASR / translation model args needed
    by JapaneseSpeechTranscriber and the HTTP translator.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Stream microphone audio, detect speech waves, then transcribe "
            "(Japanese) and translate (English) each wave on-the-fly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── tracker flags (duplicated from get_args() so we own the parser) ──
    parser.add_argument(
        "-o", "--output", dest="output_dir", default=default_output_dir, type=Path
    )
    parser.add_argument("-t", "--threshold", type=float, default=0.1)
    parser.add_argument("-c", "--close-threshold", type=float, default=None)
    parser.add_argument("--vad-interval", type=float, default=1.0)
    parser.add_argument("--max-buffer", type=float, default=30.0)
    parser.add_argument("--prob-weight", type=float, default=0.5)
    parser.add_argument("--rms-weight", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=1600)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--disable-merge", action="store_true", default=False)

    # ── ASR model flags ──
    parser.add_argument(
        "--tokens", type=str, default="", help="Path to tokens.txt for the ASR model"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        help="Path to encoder-*.int8.onnx (zipformer_reazon)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="",
        help="Path to decoder-*.int8.onnx (zipformer_reazon)",
    )
    parser.add_argument(
        "--joiner",
        type=str,
        default="",
        help="Path to joiner-*.int8.onnx (zipformer_reazon)",
    )
    parser.add_argument(
        "--sense-voice", type=str, default="", help="Path to SenseVoice model.int8.onnx"
    )
    parser.add_argument(
        "--silero-vad-model",
        type=str,
        default="",
        help="Path to silero_vad.onnx (used by JapaneseSpeechTranscriber)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["zipformer_reazon", "sense_voice"],
        default="zipformer_reazon",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--feature-dim", type=int, default=80)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    args = get_full_args(OUTPUT_DIR)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build the ASR transcriber (lazy: model loads on first use) ──────────
    console.print("[cyan]Loading JapaneseSpeechTranscriber…[/cyan]")
    transcriber = JapaneseSpeechTranscriber(
        encoder=args.encoder or None,
        decoder=args.decoder or None,
        joiner=args.joiner or None,
        sense_voice=args.sense_voice or None,
        tokens=args.tokens or None,
        silero_vad_model=args.silero_vad_model or None,
        model_type=args.model_type,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        debug=args.debug,
    )

    # ── on_wave callback: transcribe + translate + save .srt ─────────────────
    def on_wave(wave, wave_dir: Path) -> None:
        """
        Called by SpeechWavesTracker right after each valid wave is saved.

        Steps:
          1. Point at the wave's sound.wav that was already written to disk.
          2. Transcribe it → Japanese text.
          3. Translate  → English text.
          4. Save wave_dir/transcript.srt with both lines.
        """
        wav_path = wave_dir / "sound.wav"
        start_sec: float = wave["start_sec"]
        end_sec: float = wave["end_sec"]

        # — Transcribe --------------------------------------------------------
        try:
            japanese_text: str = transcriber.transcribe_japanese(str(wav_path)).strip()
        except Exception as exc:
            console.log(f"[red]Transcription error for {wav_path.name}: {exc}[/red]")
            japanese_text = ""

        # — Translate ---------------------------------------------------------
        english_text = ""
        if japanese_text:
            try:
                result = translate(japanese_text)
                english_text = result.get("text", "").strip()
            except Exception as exc:
                console.log(f"[red]Translation error: {exc}[/red]")

        # — Save SRT ----------------------------------------------------------
        srt_path = save_srt(wave_dir, start_sec, end_sec, japanese_text, english_text)

        # — Console summary ---------------------------------------------------
        console.log(
            f"[bold green]Wave {wave['start_sec']:.2f}s → {wave['end_sec']:.2f}s[/bold green]\n"
            f"  🇯🇵 [yellow]{japanese_text or '(no speech detected)'}[/yellow]\n"
            f"  🇬🇧 [cyan]{english_text or '(no translation)'}[/cyan]\n"
            f"  📄 SRT → [link=file://{srt_path.resolve()}]{srt_path.name}[/link]"
        )

    tracker = SpeechWavesTracker(
        output_dir=args.output_dir,
        sample_rate=SAMPLE_RATE,
        threshold=args.threshold,
        close_threshold=args.close_threshold,
        vad_interval_sec=args.vad_interval,
        max_buffer_sec=args.max_buffer,
        prob_weight=args.prob_weight,
        rms_weight=args.rms_weight,
        disable_merge=args.disable_merge,
        on_wave=on_wave,
    )

    # sounddevice puts chunks into this queue; the main thread drains it.
    # Using a queue decouples the real-time callback from slower VAD work.
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def sd_callback(
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice in a background thread for every block."""
        if status:
            console.log(f"[yellow]Stream status: {status}[/yellow]")
        # Copy so the buffer isn't mutated after we return
        audio_queue.put(indata[:, 0].copy())

    console.print(
        "[bold cyan]🎙  Recording"
        + (f" for {args.duration}s" if args.duration else " (Ctrl-C to stop)")
        + f" → {args.output_dir}[/bold cyan]"
    )

    start_time = time.monotonic()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=args.blocksize,
        device=args.device,
        callback=sd_callback,
    ):
        try:
            while True:
                # Drain the queue in the main thread
                try:
                    chunk = audio_queue.get(timeout=0.05)
                    tracker.feed(chunk)
                except queue.Empty:
                    pass

                elapsed = time.monotonic() - start_time
                if args.duration > 0 and elapsed >= args.duration:
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — flushing final buffer…[/yellow]")

    # Process any remaining audio in the queue
    while not audio_queue.empty():
        tracker.feed(audio_queue.get_nowait())

    # Finalize: flush buffer and save summary files
    tracker.flush()
    tracker.save_summary_files()

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{tracker._wave_counter} wave(s) saved to [cyan]{args.output_dir}[/cyan]"
    )

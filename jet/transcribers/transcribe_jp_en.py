# fast_jaen_transcriber.py
from __future__ import annotations

import torch
from pathlib import Path
from typing import Literal, Iterable, List, Dict, Any
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from transformers import pipeline, Pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
log = logging.getLogger("FastJaEnTranscriber")
console = Console()


Language = Literal["japanese", "english"]
Task = Literal["transcribe"]


class FastJaEnTranscriber:
    """
    A fast, reusable Japanese & English transcriber using Kotoba Whisper v2.0
    Optimized for both ja→ja and en→en with automatic language detection fallback.
    """

    MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
    NORMALIZER = BasicTextNormalizer()

    def __init__(
        self,
        *,
        device: str | int | None = None,
        torch_dtype: torch.dtype | None = None,
        batch_size: int = 8,
        use_flash_attention_2: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        attn_impl = "flash_attention_2" if use_flash_attention_2 and torch.cuda.is_available() else "sdpa"
        model_kwargs = {"attn_implementation": attn_impl} if attn_impl != "eager" else {}

        log.info(f"Loading model [cyan]{self.MODEL_ID}[/] → {self.device} ({self.torch_dtype})")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Initializing pipeline...", total=None)

            self.pipe: Pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_ID,
                torch_dtype=self.torch_dtype,
                device=self.device,
                model_kwargs=model_kwargs,
                batch_size=self.batch_size,
            )

        log.info("Model loaded successfully")

    def _normalize_text(self, text: str) -> str:
        """Normalize and clean transcription text (remove spaces, lowercase, etc.)"""
        return self.NORMALIZER(text).replace(" ", "")

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: Language | None = None,
        task: Task = "transcribe",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to audio file (supports wav, mp3, flac, etc.)
            language: "japanese" or "english". If None, model auto-detects (recommended).
            task: Always "transcribe"
            return_timestamps: Whether to return word-level timestamps

        Returns:
            Dict with 'text', 'normalized_text', and optionally 'chunks' with timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        generate_kwargs = {
            "task": task,
            "return_timestamps": return_timestamps,
        }
        if language:
            generate_kwargs["language"] = language

        log.info(f"Transcribing [green]{audio_path.name}[/] ({language or 'auto'})")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Transcribing..."),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Processing", total=None)
            result = self.pipe(
                str(audio_path),
                generate_kwargs=generate_kwargs,
                batch_size=self.batch_size,
            )
            progress.update(task, completed=True)

        raw_text = result.get("text", "") if isinstance(result, dict) else result
        norm_text = self._normalize_text(raw_text)

        output = {
            "text": raw_text.strip(),
            "normalized_text": norm_text,
            "language": language or "auto-detected",
        }

        if return_timestamps and "chunks" in result:
            output["chunks"] = result["chunks"]

        return output

    def transcribe_batch(
        self,
        audio_paths: Iterable[str | Path],
        language: Language | None = None,
        return_timestamps: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files efficiently with batch processing and progress bar.
        """
        paths = [Path(p) for p in audio_paths]
        results: List[Dict[str, Any]] = []

        generate_kwargs = {"task": "transcribe", "language": language} if language else {"task": "transcribe"}

        log.info(f"Batch transcribing {len(paths)} files → {language or 'auto'}")

        with Progress(
            "[progress.description]{task.description}",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Transcribing batch", total=len(paths))

            for result in self.pipe(
                [str(p) for p in paths],
                batch_size=self.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps,
            ):
                raw_text = result.get("text", "") if isinstance(result, dict) else result
                norm_text = self._normalize_text(raw_text)

                output = {
                    "file": paths[progress.tasks[task_id].completed],
                    "text": raw_text.strip(),
                    "normalized_text": norm_text,
                }
                if return_timestamps and "chunks" in result:
                    output["chunks"] = result["chunks"]

                results.append(output)
                progress.advance(task_id)

        return results

# Example usage
AUDIO_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/audio/speech/pyannote/generated/stream_speakers_extractor/speakers/004_13.23_16.64_SPEAKER_01/sound.wav"
if __name__ == "__main__":
    transcriber = FastJaEnTranscriber(batch_size=16)

    # Single file (auto language detection)
    result = transcriber.transcribe_file(AUDIO_PATH)
    print(result["text"])

    # Force Japanese
    result = transcriber.transcribe_file(AUDIO_PATH, language="japanese")
    print(result["normalized_text"])

    # # Batch transcribe a folder
    # import glob
    # files = glob.glob("audio_samples/*.wav")
    # results = transcriber.transcribe_batch(files, language="english")
    # for r in results:
    #     print(f"{r['file'].name}: {r['text'][:60]}...")

from __future__ import annotations

from enum import Enum
from pathlib import Path

import ffmpeg
from pydantic import BaseModel, Field, conint, validator
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from tqdm import tqdm


class AudioCodec(str, Enum):
    """FFmpeg codec names with CLI-friendly lookup support."""
    FLAC = "flac"
    OPUS = "libopus"
    ALAC = "alac"
    WAVPACK = "wavpack"

    @classmethod
    def _missing_(cls, value: object) -> "AudioCodec":
        """Allow CLI users to pass short names like 'opus' instead of 'libopus'."""
        value_lower = str(value).lower()
        mapping = {
            "opus": cls.OPUS,
            "flac": cls.FLAC,
            "alac": cls.ALAC,
            "wavpack": cls.WAVPACK,
        }
        if match := mapping.get(value_lower):
            return match
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")


class CompressionConfig(BaseModel):
    codec: AudioCodec = Field(default=AudioCodec.FLAC, description="Target lossless codec")
    opus_bitrate_kbps: conint(ge=256, le=512) = Field(
        default=512, description="Only used when codec=OPUS. 512 = effectively lossless"
    )
    compression_level: conint(ge=0, le=12) = Field(
        default=8, description="FLAC: 0=fastest, 12=best. WavPack: 0–3 (high=better)"
    )
    keep_original: bool = Field(default=False, description="Keep source file after conversion")
    overwrite: bool = Field(default=False, description="Overwrite existing output file")

    @validator("compression_level")
    def validate_level(cls, v, values):
        codec = values.get("codec")
        if codec == AudioCodec.OPUS:
            return 0  # ignored for opus
        if codec == AudioCodec.WAVPACK and v > 3:
            return 3
        return v


def get_output_path(input_path: Path, codec: AudioCodec) -> Path:
    suffix_map = {
        AudioCodec.FLAC: ".flac",
        AudioCodec.OPUS: ".opus",
        AudioCodec.ALAC: ".m4a",
        AudioCodec.WAVPACK: ".wv",
    }
    return input_path.with_suffix(suffix_map[codec])


def compress_audio_file(
    input_path: str | Path,
    config: CompressionConfig | None = None,
    progress: Progress | None = None,
) -> Path:
    config = config or CompressionConfig()
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    output_path = get_output_path(input_path, config.codec)

    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    task_id = None
    if progress:
        task_id = progress.add_task(f"[cyan]Compressing[/] {input_path.name}", total=None)

    stream = ffmpeg.input(str(input_path))

    kwargs: dict[str, object] = {"c:a": config.codec.value}
    if config.codec == AudioCodec.FLAC:
        kwargs["compression_level"] = config.compression_level
    elif config.codec == AudioCodec.OPUS:
        kwargs["b:a"] = f"{config.opus_bitrate_kbps}k"
    elif config.codec == AudioCodec.WAVPACK:
        # -hx = high, -x1–3 = extra processing
        kwargs["compression_level"] = min(config.compression_level, 3)
        if config.compression_level >= 2:
            kwargs["x"] = config.compression_level - 1

    try:
        ffmpeg.output(stream, str(output_path), **kwargs).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed: {error_msg}") from e
    finally:
        if progress and task_id:
            progress.update(task_id, completed=True)

    original_size = input_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    ratio = (1 - new_size / original_size) * 100

    rprint(
        f"[green]✓[/green] {input_path.name} → {output_path.name} "
        f"({original_size:.1f} MB → {new_size:.1f} MB, [bold green]{ratio:.1f}% smaller[/bold green])"
    )

    if not config.keep_original:
        input_path.unlink()

    return output_path


def compress_audio_folder(
    folder: Path | str,
    config: CompressionConfig | None = None,
    pattern: str = "*.*",
) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    supported_exts = {".wav", ".aiff", ".aif", ".flac", ".m4a", ".wv", ".opus"}
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in supported_exts]

    if not files:
        rprint("[yellow]No supported audio files found[/yellow]")
        return []

    config = config or CompressionConfig()

    table = Table(title="Compression Summary")
    table.add_column("File")
    table.add_column("Original → New")
    table.add_column("Savings")

    results: list[Path] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        for file in tqdm(files, desc="Compressing audio files", unit="file"):
            try:
                out_path = compress_audio_file(file, config, progress)
                results.append(out_path)
            except Exception as e:
                rprint(f"[red]✗ Failed {file.name}: {e}[/red]")

    return results

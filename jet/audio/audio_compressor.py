from __future__ import annotations

from enum import Enum
from pathlib import Path

import ffmpeg
from pydantic import BaseModel, Field, conint, validator
from rich import print as rprint
from rich.progress import Progress
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


def compress_audio(
    path: str | Path,
    *,
    config: CompressionConfig | None = None,
    recursive: bool = False,
    output_dir: Path | str | None = None,
    progress: Progress | None = None,
) -> list[Path]:
    """
    Unified, pure compression function.

    - Works on a single file OR a directory (optionally recursive)
    - If ``output_dir`` is given  → all outputs go there with ``stem + proper suffix``
    - If ``output_dir`` is None → output is placed next to input (original behavior)
    """
    config = config or CompressionConfig()
    root = Path(path).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve() if output_dir else None

    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    supported = {".wav", ".aiff", ".aif", ".flac", ".m4a", ".wv", ".opus"}

    # Collect files
    if root.is_file():
        if root.suffix.lower() not in supported:
            raise ValueError(f"Unsupported audio format: {root}")
        files = [root]
    elif root.is_dir():
        iterator = root.rglob("*") if recursive else root.iterdir()
        files = [p for p in iterator if p.is_file() and p.suffix.lower() in supported]
        if not files:
            rprint("[yellow]No supported audio files found in directory[/yellow]")
            return []
    else:
        raise FileNotFoundError(f"Path not found: {root}")

    results: list[Path] = []
    task_id = None
    if progress and len(files) > 1:
        task_id = progress.add_task("[cyan]Compressing…", total=len(files))

    suffix_map = {
        AudioCodec.FLAC: ".flac",
        AudioCodec.OPUS: ".opus",
        AudioCodec.ALAC: ".m4a",
        AudioCodec.WAVPACK: ".wv",
    }
    target_suffix = suffix_map[config.codec]

    for input_path in tqdm(files, desc="Compressing", unit="file", disable=len(files) == 1):
        # Determine output path
        if output_dir_path:
            output_path = output_dir_path / f"{input_path.stem}{target_suffix}"
        else:
            output_path = input_path.with_suffix(target_suffix)

        if output_path.exists() and not config.overwrite:
            rprint(f"[dim]Skipping (exists)[/] {output_path.name}")
            results.append(output_path)
            continue

        # Build ffmpeg args
        stream = ffmpeg.input(str(input_path))
        kwargs: dict[str, object] = {"c:a": config.codec.value}

        if config.codec == AudioCodec.FLAC:
            kwargs["compression_level"] = config.compression_level
        elif config.codec == AudioCodec.OPUS:
            kwargs["b:a"] = f"{config.opus_bitrate_kbps}k"
        elif config.codec == AudioCodec.WAVPACK:
            level = min(config.compression_level, 3)
            kwargs["compression_level"] = level
            if level >= 2:
                kwargs["x"] = level - 1

        try:
            ffmpeg.output(stream, str(output_path), **kwargs).overwrite_output().run(
                capture_stdout=True, capture_stderr=True
            )
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if e.stderr else str(e)
            rprint(f"[red]Failed[/] {input_path.name}: {msg}")
            continue

        # Stats + optional delete
        orig_mb = input_path.stat().st_size / (1024 * 1024)
        new_mb = output_path.stat().st_size / (1024 * 1024)
        savings = (1 - new_mb / orig_mb) * 100

        rprint(
            f"[green]Success[/] {input_path.name} → {output_path.name} "
            f"({orig_mb:.1f} → {new_mb:.1f} MB, [bold green]{savings:.1f}%[/] saved)"
        )

        if not config.keep_original:
            input_path.unlink()
            rprint("[dim]Original deleted[/]")

        results.append(output_path)

    if progress and task_id:
        progress.update(task_id, completed=True)

    return results

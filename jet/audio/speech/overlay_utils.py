from pathlib import Path

from jet.audio.helpers.silence import SAMPLE_RATE
from jet.logger import logger

def _samples_to_timestamp(samples: int) -> str:
    """Convert sample count to SRT timestamp format HH:MM:SS,mmm"""
    seconds_total = samples / SAMPLE_RATE
    hours = int(seconds_total // 3600)
    minutes = int((seconds_total % 3600) // 60)
    seconds = int(seconds_total % 60)
    millis = int((seconds_total - int(seconds_total)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def write_srt_file(
    filepath: Path,
    text: str,
    start_sample: int,
    end_sample: int,
    index: int = 1,
) -> None:
    """
    Write a single-entry SRT file.
    """
    content = f"{index}\n"
    content += f"{_samples_to_timestamp(start_sample)} --> {_samples_to_timestamp(end_sample)}\n"
    content += f"{text.strip()}\n\n"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"SRT saved → {filepath}")

def append_to_combined_srt(
    combined_path: Path,
    text: str,
    start_sample: int,
    end_sample: int,
    index: int,
) -> None:
    """
    Append a single subtitle entry to the combined all_subtitles.srt.
    """
    content = f"{index}\n"
    content += f"{_samples_to_timestamp(start_sample)} --> {_samples_to_timestamp(end_sample)}\n"
    content += f"{text.strip()}\n\n"
    with combined_path.open("a", encoding="utf-8") as f:
        f.write(content)
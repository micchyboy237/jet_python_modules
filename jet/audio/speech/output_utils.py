from pathlib import Path

from jet.audio.helpers.silence import SAMPLE_RATE
from jet.logger import logger

def _samples_to_timestamp(samples: int) -> str:
    # Convert absolute sample count to total seconds from recording start
    total_seconds = samples / SAMPLE_RATE

    # Extract hours, minutes, seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Extract milliseconds (fractional part)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))

    # Ensure millisecond is exactly 3 digits
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def write_srt_file(
    filepath: Path,
    source_text: str,
    target_text: str,
    start_sample: float,  # now in seconds from start of recording
    end_sample: float,    # now in seconds from start of recording
    index: int = 1,
) -> None:
    """
    Write a single-entry SRT file with optional bilingual subtitles.
    If target_text is provided, it will be displayed below source_text.
    """
    content = f"{index}\n"
    content += f"{_seconds_to_timestamp(start_sample)} --> {_seconds_to_timestamp(end_sample)}\n"

    if target_text.strip():
        content += f"{source_text.strip()}\n{target_text.strip()}\n\n"
    else:
        content += f"{source_text.strip()}\n\n"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"SRT saved → {filepath}")

def append_to_combined_srt(
    combined_path: Path,
    source_text: str,
    target_text: str,
    start_sample: float,  # now in seconds from start of recording
    end_sample: float,    # now in seconds from start of recording
    index: int,
) -> None:
    """
    Append a single subtitle entry to the combined all_subtitles.srt.
    If target_text is provided, it will be displayed below source_text.
    """
    content = f"{index}\n"
    content += f"{_seconds_to_timestamp(start_sample)} --> {_seconds_to_timestamp(end_sample)}\n"

    if target_text.strip():
        content += f"{source_text.strip()}\n{target_text.strip()}\n\n"
    else:
        content += f"{source_text.strip()}\n\n"

    with combined_path.open("a", encoding="utf-8") as f:
        f.write(content)

def _seconds_to_timestamp(seconds: float) -> str:
    """Convert absolute seconds (from recording start) to SRT timestamp HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
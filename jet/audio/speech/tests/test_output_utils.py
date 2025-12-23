from pathlib import Path
from typing import Final
import pytest
from jet.audio.speech.output_utils import (
    _seconds_to_timestamp,
    write_srt_file,
    append_to_combined_srt,
)

# Real-world example timings based on the user's recorded segments
TEST_CASES: Final[list[dict]] = [
    {
        "index": 1,
        "ja_text": "大丈夫ですか?",
        "en_text": "Are you okay?",
        "start_sec": 4.450,
        "end_sec": 5.860,
        "expected_timestamp_line": "00:00:04,450 --> 00:00:05,860",
    },
    {
        "index": 2,
        "ja_text": "本当にいつも来ていただいて。",
        "en_text": "I've always been there.",
        "start_sec": 8.290,
        "end_sec": 10.300,
        "expected_timestamp_line": "00:00:08,290 --> 00:00:10,300",
    },
]

@pytest.fixture(scope="function")
def temp_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary directory for each test."""
    return tmp_path


def test__seconds_to_timestamp() -> None:
    """Given various second values, _seconds_to_timestamp should produce correct SRT format."""
    # Given
    test_data = [
        (0.0, "00:00:00,000"),
        (4.45, "00:00:04,450"),
        (5.86, "00:00:05,860"),
        (8.29, "00:00:08,290"),
        (10.30, "00:00:10,300"),
        (59.999, "00:00:59,999"),
        (60.0, "00:01:00,000"),
        (3661.123, "01:01:01,123"),
    ]

    # When / Then
    for seconds, expected in test_data:
        result = _seconds_to_timestamp(seconds)
        assert result == expected, f"Failed for {seconds}s: got {result}, expected {expected}"


@pytest.mark.parametrize("case", TEST_CASES)
def test_write_srt_file(temp_dir: Path, case: dict) -> None:
    """Given real-world segment data, write_srt_file should create correct single-entry SRT."""
    # Given
    filepath = temp_dir / "subtitles.srt"
    index = case["index"]
    ja_text = case["ja_text"]
    en_text = case["en_text"]
    start_sec = case["start_sec"]
    end_sec = case["end_sec"]
    expected_timestamp_line = case["expected_timestamp_line"]

    expected_content = (
        f"{index}\n"
        f"{expected_timestamp_line}\n"
        f"{ja_text}\n"
        f"{en_text}\n"
        "\n"
    )

    # When
    write_srt_file(
        filepath=filepath,
        source_text=ja_text,
        target_text=en_text,
        start_sample=start_sec,
        end_sample=end_sec,
        index=index,
    )

    # Then
    result_content = filepath.read_text(encoding="utf-8")
    assert result_content == expected_content
    assert filepath.exists()


@pytest.mark.parametrize("case", TEST_CASES)
def test_append_to_combined_srt(temp_dir: Path, case: dict) -> None:
    """Given multiple segments, append_to_combined_srt should build a correct combined SRT."""
    # Given
    combined_path = temp_dir / "all_subtitles.srt"
    index = case["index"]
    ja_text = case["ja_text"]
    en_text = case["en_text"]
    start_sec = case["start_sec"]
    end_sec = case["end_sec"]
    expected_timestamp_line = case["expected_timestamp_line"]

    expected_entry = (
        f"{index}\n"
        f"{expected_timestamp_line}\n"
        f"{ja_text}\n"
        f"{en_text}\n"
        "\n"
    )

    # When
    append_to_combined_srt(
        combined_path=combined_path,
        source_text=ja_text,
        target_text=en_text,
        start_sample=start_sec,
        end_sample=end_sec,
        index=index,
    )

    # Then
    result_content = combined_path.read_text(encoding="utf-8")
    assert result_content == expected_entry


def test_append_to_combined_srt_multiple_segments(temp_dir: Path) -> None:
    """Given two real-world segments in order, the combined SRT should contain both entries correctly."""
    # Given
    combined_path = temp_dir / "all_subtitles.srt"

    expected_content = ""
    for case in TEST_CASES:
        expected_content += (
            f"{case['index']}\n"
            f"{case['expected_timestamp_line']}\n"
            f"{case['ja_text']}\n"
            f"{case['en_text']}\n"
            "\n"
        )

    # When
    for case in TEST_CASES:
        append_to_combined_srt(
            combined_path=combined_path,
            source_text=case["ja_text"],
            target_text=case["en_text"],
            start_sample=case["start_sec"],
            end_sample=case["end_sec"],
            index=case["index"],
        )

    # Then
    result_content = combined_path.read_text(encoding="utf-8")
    assert result_content == expected_content
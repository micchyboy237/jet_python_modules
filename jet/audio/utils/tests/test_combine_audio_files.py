# tests/test_combine_audio_files.py

import tempfile
from pathlib import Path

import pytest
from jet.audio.utils.combine import combine_audio_files
from pydub import AudioSegment


@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path

    # Cleanup
    if dir_path.exists():
        for item in dir_path.iterdir():
            item.unlink()
        dir_path.rmdir()


@pytest.fixture
def output_path(temp_dir: Path):
    return temp_dir / "combined.wav"


def create_silent_audio(
    base_dir: Path,
    filename: str,
    duration_ms: int = 1000,
    sample_rate: int = 44100,
    channels: int = 1,
) -> Path:
    """
    Create silent WAV with configurable channels.
    """
    path = base_dir / filename
    silent = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
    silent = silent.set_channels(channels)
    silent.export(str(path), format="wav")
    return path


def test_concatenate_two_files_default_settings(temp_dir: Path, output_path: Path):
    # Given
    file1 = create_silent_audio(temp_dir, "test1.wav", 1500)
    file2 = create_silent_audio(temp_dir, "test2.wav", 2000)

    # When
    combine_audio_files([file1, file2], output_path)

    # Then
    result = AudioSegment.from_wav(str(output_path))
    expected = 3500

    assert len(result) == expected
    assert result.frame_rate == 16000
    assert result.sample_width == 2
    assert result.channels == 1


def test_channel_strategy_mono_forces_downmix(temp_dir: Path, output_path: Path):
    # Given
    stereo = create_silent_audio(temp_dir, "stereo.wav", 1000, channels=2)

    # When
    combine_audio_files(
        [stereo],
        output_path,
        channel_strategy="mono",
    )

    # Then
    result = AudioSegment.from_wav(str(output_path))
    expected = 1

    assert result.channels == expected


def test_channel_strategy_stereo_upmixes(temp_dir: Path, output_path: Path):
    # Given
    mono = create_silent_audio(temp_dir, "mono.wav", 1000, channels=1)

    # When
    combine_audio_files(
        [mono],
        output_path,
        channel_strategy="stereo",
    )

    # Then
    result = AudioSegment.from_wav(str(output_path))
    expected = 2

    assert result.channels == expected


def test_channel_strategy_match_first(temp_dir: Path, output_path: Path):
    # Given
    stereo = create_silent_audio(temp_dir, "stereo.wav", channels=2)
    mono = create_silent_audio(temp_dir, "mono.wav", channels=1)

    # When
    combine_audio_files(
        [stereo, mono],
        output_path,
        channel_strategy="match-first",
    )

    # Then
    result = AudioSegment.from_wav(str(output_path))
    expected = 2

    assert result.channels == expected


def test_concatenate_three_files_different_lengths(temp_dir: Path, output_path: Path):
    # Given
    f1 = create_silent_audio(temp_dir, "a.wav", 800)
    f2 = create_silent_audio(temp_dir, "b.wav", 1200)
    f3 = create_silent_audio(temp_dir, "c.wav", 500)

    # When
    combine_audio_files([f1, f2, f3], output_path)

    # Then
    result = AudioSegment.from_wav(str(output_path))
    expected = 2500

    assert len(result) == expected
    assert result.frame_rate == 16000
    assert result.sample_width == 2


def test_concatenate_with_int32_dtype(temp_dir: Path):
    # Given
    file1 = create_silent_audio(temp_dir, "test.wav", 1000)
    output = temp_dir / "combined_int32.wav"

    # When
    combine_audio_files([file1], output, dtype="int32")

    # Then
    result = AudioSegment.from_wav(str(output))
    expected_width = 4

    assert result.sample_width == expected_width
    assert result.frame_rate == 16000


def test_no_input_files_raises_error(output_path: Path):
    # Given / When / Then
    with pytest.raises(ValueError):
        combine_audio_files([], output_path)


def test_input_file_not_found(temp_dir: Path, output_path: Path):
    # Given
    real_file = create_silent_audio(temp_dir, "real.wav")
    fake_file = temp_dir / "missing.wav"

    # When / Then
    with pytest.raises(FileNotFoundError):
        combine_audio_files([real_file, fake_file], output_path)


def test_output_directory_is_created(temp_dir: Path):
    # Given
    deep_output = temp_dir / "sub" / "deep" / "combined.wav"
    silent = create_silent_audio(temp_dir, "test.wav", 500)

    # When
    combine_audio_files([silent], deep_output)

    # Then
    assert deep_output.parent.exists()
    assert deep_output.exists()

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jet.audio.audio_compressor import (
    AudioCodec,
    CompressionConfig,
    compress_audio_file,
    compress_audio_folder,
    get_output_path,
)

TEST_DIR = Path(__file__).parent / "test_samples"
TEST_DIR.mkdir(exist_ok=True)
SAMPLE_WAV = TEST_DIR / "sine_1khz.wav"


@pytest.fixture(scope="session")
def sample_wav():
    # Generate a 1-second 44.1kHz stereo sine wave (deterministic)
    if not SAMPLE_WAV.exists():
        import ffmpeg
        ffmpeg.input("anullsrc=channel_layout=stereo:sample_rate=44100", t=1, f="lavfi").output(
            str(SAMPLE_WAV), acodec="pcm_s16le"
        ).overwrite_output().run(quiet=True)
    return SAMPLE_WAV


def test_get_output_path():
    # Given
    input_path = Path("test.wav")
    # When / Then
    assert get_output_path(input_path, AudioCodec.FLAC) == Path("test.flac")
    assert get_output_path(input_path, AudioCodec.OPUS) == Path("test.opus")
    assert get_output_path(input_path, AudioCodec.ALAC) == Path("test.m4a")


def test_compress_flac_lossless(sample_wav, tmp_path):
    # Given
    input_file = shutil.copy2(sample_wav, tmp_path / "input.wav")
    config = CompressionConfig(codec=AudioCodec.FLAC, compression_level=8, keep_original=True)

    # When
    output_file = compress_audio_file(input_file, config)

    # Then
    expected_output = get_output_path(input_file, AudioCodec.FLAC)
    assert output_file == expected_output
    assert output_file.exists()
    assert input_file.exists()  # kept
    assert output_file.stat().st_size < input_file.stat().st_size * 0.7  # at least 30% smaller


def test_compress_opus_high_quality(sample_wav, tmp_path):
    # Given
    input_file = shutil.copy2(sample_wav, tmp_path / "input.wav")
    config = CompressionConfig(codec=AudioCodec.OPUS, opus_bitrate_kbps=512, keep_original=True)

    # When
    output_file = compress_audio_file(input_file, config)

    # Then
    expected_output = get_output_path(input_file, AudioCodec.OPUS)
    assert output_file == expected_output
    assert output_file.stat().st_size < input_file.stat().st_size * 0.5  # usually >50% smaller


def test_compress_folder_no_files(tmp_path):
    # Given
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    # When
    results = compress_audio_folder(empty_folder)

    # Then
    expected = []
    assert results == expected


def test_compress_folder_with_files(sample_wav, tmp_path):
    # Given
    folder = tmp_path / "audio_folder"
    folder.mkdir()
    shutil.copy2(sample_wav, folder / "track1.wav")
    shutil.copy2(sample_wav, folder / "track2.aiff")

    config = CompressionConfig(codec=AudioCodec.FLAC, keep_original=False)

    # When
    results = compress_audio_folder(folder, config)

    # Then
    expected = [
        folder / "track1.flac",
        folder / "track2.flac",
    ]
    assert sorted(results) == sorted(expected)
    assert not (folder / "track1.wav").exists()
    assert not (folder / "track2.aiff").exists()
# jet_python_modules/jet/audio/speech/tests/test_resolve_audio_paths.py

from pathlib import Path
from typing import List

import pytest
from jet.audio.utils import resolve_audio_paths, AUDIO_EXTENSIONS


# ──────────────────────────────────────────────────────────────
# Helper to create fake audio files
# ──────────────────────────────────────────────────────────────
def create_fake_audio_files(tmp_path: Path, filenames: List[str]) -> None:
    for name in filenames:
        (tmp_path / name).write_bytes(b"RIFF fake wav" if name.endswith(".wav") else b"fake audio data")


# ──────────────────────────────────────────────────────────────
# BDD-style tests – no caplog, only return values & exceptions
# ──────────────────────────────────────────────────────────────
class TestResolveAudioPaths:
    def test_given_single_existing_audio_file_returns_list_with_that_path(self, tmp_path: Path):
        # Given
        audio_file = tmp_path / "speech.wav"
        audio_file.write_bytes(b"fake wav")
        expected = [audio_file]

        # When
        result = resolve_audio_paths(str(audio_file))

        # Then
        assert result == expected

    def test_given_path_object_works_the_same_as_string(self, tmp_path: Path):
        # Given
        audio_file = tmp_path / "music.mp3"
        audio_file.touch()
        expected = [audio_file]

        # When
        result_str = resolve_audio_paths(str(audio_file))
        result_path = resolve_audio_paths(audio_file)

        # Then
        assert result_str == expected
        assert result_path == expected
        assert result_str == result_path

    def test_given_directory_with_supported_audio_files_returns_all_of_them(self, tmp_path: Path):
        # Given
        supported = ["a.wav", "b.mp3", "c.flac", "deep.m4a"]
        create_fake_audio_files(tmp_path, supported + ["ignore.txt", "video.mp4"])
        expected = sorted(tmp_path / f for f in supported)

        # When
        result = resolve_audio_paths(tmp_path)

        # Then
        assert sorted(result) == expected

    def test_given_empty_directory_raises_valueerror(self, tmp_path: Path):
        # Given - empty directory (no files at all)
        # When / Then
        with pytest.raises(ValueError, match="No valid audio files found"):
            resolve_audio_paths(tmp_path)

    def test_given_directory_with_only_non_audio_files_raises_valueerror(self, tmp_path: Path):
        # Given
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "image.jpg").touch()

        # When / Then
        with pytest.raises(ValueError, match="No valid audio files found"):
            resolve_audio_paths(tmp_path)

    def test_given_list_of_mixed_inputs_returns_only_valid_audio_files(self, tmp_path: Path):
        # Given
        wav1 = tmp_path / "one.wav"
        wav2 = tmp_path / "two.wav"
        wav1.touch()
        wav2.touch()

        dir_with_audio = tmp_path / "folder"
        dir_with_audio.mkdir()
        (dir_with_audio / "three.m4a").touch()

        unsupported = tmp_path / "doc.pdf"
        unsupported.touch()

        missing = tmp_path / "ghost.flac"

        inputs = [
            str(wav1),
            wav2,
            str(dir_with_audio),
            unsupported,
            str(missing),
        ]

        expected = sorted([wav1, wav2, dir_with_audio / "three.m4a"])

        # When
        result = resolve_audio_paths(inputs)

        # Then
        assert sorted(result) == expected

    def test_given_no_valid_inputs_at_all_raises_valueerror(self, tmp_path: Path):
        # Given
        bad_file = tmp_path / "data.txt"
        bad_file.touch()
        missing = tmp_path / "nope.wav"
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # When / Then
        with pytest.raises(ValueError, match="No valid audio files found"):
            resolve_audio_paths([str(bad_file), missing, empty_dir])

    def test_case_insensitive_extension_matching(self, tmp_path: Path):
        # Given
        files = ["UPPER.WAV", "Mixed.Mp3", "lower.ogg"]
        create_fake_audio_files(tmp_path, files)
        expected = sorted(tmp_path / f for f in files)

        # When
        result = resolve_audio_paths(tmp_path)

        # Then
        assert sorted(result) == expected

    def test_non_audio_files_are_skipped_and_do_not_appear_in_result(self, tmp_path: Path):
        # Given
        (tmp_path / "song.wav").touch()
        (tmp_path / "image.jpg").touch()
        (tmp_path / "video.mkv").touch()  # .mkv is supported → must be included
        expected = sorted([tmp_path / "song.wav", tmp_path / "video.mkv"])

        # When
        result = resolve_audio_paths(tmp_path)

        # Then
        assert sorted(result) == expected

    def test_missing_path_is_ignored_but_does_not_prevent_success_if_other_valid_files_exist(self, tmp_path: Path):
        # Given
        real_file = tmp_path / "exists.flac"
        real_file.touch()
        missing_file = tmp_path / "does_not_exist.wav"

        # When
        result = resolve_audio_paths([real_file, missing_file])

        # Then
        assert result == [real_file]

    def test_completely_missing_single_input_raises_valueerror(self, tmp_path: Path):
        # Given
        missing = tmp_path / "does_not_exist.wav"

        # When / Then
        with pytest.raises(ValueError, match="No valid audio files found"):
            resolve_audio_paths(str(missing))

    def test_given_recursive_true_scans_subdirectories(self, tmp_path: Path):
        # Given
        deep_dir = tmp_path / "outer" / "inner" / "deepest"
        deep_dir.mkdir(parents=True, exist_ok=True)

        top_level = tmp_path / "top.wav"
        mid_level = tmp_path / "outer" / "mid.m4a"
        deep_file = deep_dir / "deep.flac"

        top_level.touch()
        mid_level.touch()
        deep_file.touch()

        # non-audio files are ignored
        (tmp_path / "outer" / "ignore.txt").touch()

        expected = sorted([top_level, mid_level, deep_file])

        # When
        result_recursive = resolve_audio_paths(tmp_path, recursive=True)
        result_non_recursive = resolve_audio_paths(tmp_path, recursive=False)

        # Then
        assert sorted(result_recursive) == expected
        assert result_non_recursive == [top_level]  # only top-level when recursive=False

    def test_results_are_sorted_by_absolute_path(self, tmp_path: Path):
        # Given
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        file1 = dir_b / "zzz.wav"
        file2 = dir_a / "aaa.wav"
        file1.touch()
        file2.touch()

        # Intentionally unsorted input order
        inputs = [file1, file2]

        expected = sorted([file2, file1], key=lambda p: p.resolve())

        # When
        result = resolve_audio_paths(inputs)

        # Then
        assert result == expected


# ──────────────────────────────────────────────────────────────
# Parametrized check for every supported extension
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ext", sorted(AUDIO_EXTENSIONS))
def test_all_supported_extensions_are_accepted(tmp_path: Path, ext):
    # Given
    audio_file = tmp_path / f"test{ext}"
    audio_file.touch()
    expected = [audio_file]

    # When
    result = resolve_audio_paths(audio_file)

    # Then
    assert result == expected
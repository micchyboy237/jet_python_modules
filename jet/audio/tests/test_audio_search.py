import tempfile
import io
from pathlib import Path
from typing import List
from typing import TypedDict
import pytest
import torch
import torchaudio
from chromadb.api.types import QueryResult
from jet.audio.audio_search import (
    AudioSegmentDatabase,
    load_audio_segment,
)


class ExpectedMetadata(TypedDict):
    file: str
    start_sec: float
    end_sec: float


# Fixture: Create a temporary database and clean up afterwards
@pytest.fixture
def temp_db() -> AudioSegmentDatabase:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = AudioSegmentDatabase(persist_dir=tmpdir)
        yield db


# Fixture: Generate simple synthetic audio segments (sine waves at different frequencies)
@pytest.fixture
def synth_audio_files() -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)

        # 440 Hz tone (A4)
        waveform_440 = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_440 = dir_path / "tone_440.wav"
        torchaudio.save(path_440, waveform_440.unsqueeze(0), 48000)

        # 880 Hz tone (A5)
        waveform_880 = torch.sin(2 * torch.pi * 880 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_880 = dir_path / "tone_880.wav"
        torchaudio.save(path_880, waveform_880.unsqueeze(0), 48000)

        # Duplicate 440 Hz
        path_440_duplicate = dir_path / "tone_440_duplicate.wav"
        torchaudio.save(path_440_duplicate, waveform_440.unsqueeze(0), 48000)

        yield {
            "similar1": str(path_440),
            "similar2": str(path_440_duplicate),
            "different": str(path_880),
        }


class TestLoadAudioSegment:
    def test_loads_file_and_returns_correct_shape(self, synth_audio_files):
        audio_path = synth_audio_files["similar1"]
        expected_samples = int(48000 * 30.0)
        waveform = load_audio_segment(audio_path, start_sec=0.0, duration_sec=30.0)
        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 1
        assert len(waveform) == expected_samples
        assert waveform.dtype == torch.float32

    def test_handles_raw_bytes_input(self, synth_audio_files):
        audio_path = synth_audio_files["similar1"]
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        expected_samples = int(48000 * 10.0)
        waveform = load_audio_segment(audio_bytes, duration_sec=10.0)
        assert isinstance(waveform, torch.Tensor)
        assert len(waveform) == expected_samples

    def test_pads_short_segment(self):
        short_wave = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 0.5, steps=int(48000 * 0.5)))
        short_bytes_io = io.BytesIO()
        torchaudio.save(short_bytes_io, short_wave.unsqueeze(0), 48000, format="wav")
        short_bytes = short_bytes_io.getvalue()
        expected_samples = int(48000 * 30.0)
        waveform = load_audio_segment(short_bytes, duration_sec=30.0)
        assert len(waveform) == expected_samples
        assert torch.all(waveform[int(48000 * 0.5):] == 0)


class TestAudioSegmentDatabase:
    def test_add_segments_stores_multiple_segments(self, temp_db, synth_audio_files):
        # Given
        audio_file = synth_audio_files["similar1"]
        expected_file_name = Path(audio_file).name

        # When
        temp_db.add_segments(
            audio_file,
            audio_name=expected_file_name,
            segment_duration_sec=30.0,
            overlap_sec=0.0
        )

        # Then
        count = temp_db.collection.count()
        assert count >= 1

        results: QueryResult = temp_db.collection.get(include=["metadatas"])
        metadata = results["metadatas"][0]
        assert Path(metadata["file"]).resolve() == Path(audio_file).resolve()
        assert metadata["start_sec"] == 0.0

    def test_add_segments_bytes_whole_file_mode(self, temp_db, synth_audio_files):
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()

        expected_id = "custom_name_full"
        expected_file_meta = "custom_name"  # Now uses audio_name
        expected_start_sec = 0.0
        expected_end_sec_approx = 2.0

        # When
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="custom_name",
            segment_duration_sec=None,
        )

        # Then
        result_count = temp_db.collection.count()
        assert result_count == 1

        results = temp_db.collection.get(include=["metadatas", "embeddings"])
        result_id = results["ids"][0]
        result_meta = results["metadatas"][0]

        assert result_id == expected_id
        assert result_meta["file"] == expected_file_meta
        assert result_meta["start_sec"] == expected_start_sec
        assert abs(result_meta["end_sec"] - expected_end_sec_approx) < 0.2

    def test_add_segments_bytes_chunked_mode(self, temp_db, synth_audio_files):
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()

        segment_duration_sec = 1.0
        overlap_sec = 0.5
        expected_min_segments = 3

        # When
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="in_memory_test",
            segment_duration_sec=segment_duration_sec,
            overlap_sec=overlap_sec,
        )

        # Then
        result_count = temp_db.collection.count()
        assert result_count >= expected_min_segments

        results = temp_db.collection.get(include=["metadatas"])
        ids = results["ids"]
        metadatas = results["metadatas"]

        expected_id_prefixes = [f"in_memory_test_{start:.1f}" for start in [0.0, 0.5, 1.0]]
        for prefix in expected_id_prefixes:
            assert any(id.startswith(prefix) for id in ids)

        start_secs = [m["start_sec"] for m in metadatas]
        assert all(start_secs[i] <= start_secs[i + 1] for i in range(len(start_secs) - 1))

        for meta in metadatas:
            assert meta["file"] == "in_memory_test"  # Uses audio_name

    def test_search_finds_bytes_added_segment(self, temp_db, synth_audio_files):
        file_path = synth_audio_files["similar1"]
        duplicate_path = synth_audio_files["similar2"]

        # Add via file path
        temp_db.add_segments(
            file_path,
            audio_name=Path(file_path).name,
            segment_duration_sec=None
        )

        # Add identical audio via bytes
        with open(duplicate_path, "rb") as f:
            audio_bytes = f.read()
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="bytes_duplicate",
            segment_duration_sec=None,
        )

        assert temp_db.collection.count() == 2

        # Search with bytes
        results = temp_db.search_similar(audio_bytes, top_k=2, duration_sec=None)

        assert len(results) == 2
        scores = [r["score"] for r in results]
        assert all(score > 0.85 for score in scores)

        files = {r["file"] for r in results}
        expected_files = {file_path, "bytes_duplicate"}  # Updated expectation
        assert {Path(f).resolve() for f in files} == {Path(f).resolve() for f in expected_files}

        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_search_returns_self_as_most_similar(self, temp_db, synth_audio_files):
        file1 = synth_audio_files["similar1"]
        file2 = synth_audio_files["similar2"]
        file_diff = synth_audio_files["different"]

        temp_db.add_segments(file1, audio_name=Path(file1).name, segment_duration_sec=None)
        temp_db.add_segments(file2, audio_name=Path(file2).name, segment_duration_sec=None)
        temp_db.add_segments(file_diff, audio_name=Path(file_diff).name, segment_duration_sec=None)

        assert temp_db.collection.count() == 3

        results = temp_db.search_similar(file1, top_k=3, duration_sec=None)

        top_file_name = Path(results[0]["file"]).name
        assert "440" in top_file_name

        assert results[0]["score"] > 0.9

        tone_list = ["440" if "440" in Path(r["file"]).name else "880" for r in results]
        assert tone_list.count("440") >= 2
        assert tone_list[-1] == "880"

        scores = [r["score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_search_with_bytes_query_works(self, temp_db, synth_audio_files):
        temp_db.add_segments(
            synth_audio_files["similar1"],
            audio_name=Path(synth_audio_files["similar1"]).name,
            segment_duration_sec=None
        )
        temp_db.add_segments(
            synth_audio_files["different"],
            audio_name=Path(synth_audio_files["different"]).name,
            segment_duration_sec=None
        )

        assert temp_db.collection.count() == 2

        with open(synth_audio_files["similar2"], "rb") as f:
            query_bytes = f.read()

        results = temp_db.search_similar(query_bytes, top_k=2, duration_sec=None)

        top_file_name = Path(results[0]["file"]).name
        assert "440" in top_file_name

        assert results[0]["score"] > 0.65

        tone_list = ["440" if "440" in Path(r["file"]).name else "880" for r in results]
        assert tone_list[0] == "440"
        assert tone_list[-1] == "880"

        scores = [r["score"] for r in results]
        assert scores[0] >= scores[1]

    def test_search_similar_handles_empty_database(self, temp_db, synth_audio_files):
        assert temp_db.collection.count() == 0
        query_path = synth_audio_files["similar1"]
        results = temp_db.search_similar(query_path, top_k=5)
        assert results == []

    def test_search_similar_handles_empty_results_but_non_empty_db(self, temp_db, synth_audio_files):
        different_file = synth_audio_files["different"]
        temp_db.add_segments(
            different_file,
            audio_name=Path(different_file).name,
            segment_duration_sec=30.0,
            overlap_sec=0.0
        )

        assert temp_db.collection.count() == 1

        query_path = synth_audio_files["similar1"]
        results = temp_db.search_similar(query_path, top_k=10)

        assert len(results) == 1
        assert "880" in Path(results[0]["file"]).name

    def test_print_results_handles_no_results(self, temp_db, capsys):
        empty_results: List[dict] = []
        temp_db.print_results(empty_results)
        captured = capsys.readouterr()
        assert "No results to display" in captured.out

    def test_add_segments_whole_file_mode(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        expected_file = audio_file

        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).name,
            segment_duration_sec=None
        )

        assert temp_db.collection.count() == 1

        results = temp_db.collection.get(include=["metadatas"])
        metadata = results["metadatas"][0]

        assert Path(metadata["file"]).resolve() == Path(expected_file).resolve()
        assert metadata["start_sec"] == 0.0
        assert abs(metadata["end_sec"] - 2.0) < 0.2

    def test_add_segments_fixed_chunk_mode(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).name,
            segment_duration_sec=1.0,
            overlap_sec=0.5
        )

        assert temp_db.collection.count() >= 2

        results = temp_db.collection.get(include=["metadatas"])
        start_secs = [m["start_sec"] for m in results["metadatas"]]
        assert all(start_secs[i] <= start_secs[i + 1] for i in range(len(start_secs) - 1))

    def test_search_similar_auto_duration_file_query(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).name,
            segment_duration_sec=None
        )

        results = temp_db.search_similar(audio_file, top_k=5, duration_sec=None)

        assert len(results) == 1
        assert results[0]["score"] > 0.9
        assert Path(results[0]["file"]).name == Path(audio_file).name

    def test_search_similar_auto_duration_bytes_fallback(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).name,
            segment_duration_sec=None
        )

        with open(audio_file, "rb") as f:
            query_bytes = f.read()

        results = temp_db.search_similar(query_bytes, top_k=5, duration_sec=None)

        assert len(results) == 1
        assert results[0]["score"] > 0.5

    def test_search_similar_returns_normalized_score(self, temp_db, synth_audio_files):
        temp_db.add_segments(
            synth_audio_files["similar1"],
            audio_name=Path(synth_audio_files["similar1"]).name,
            segment_duration_sec=None
        )
        temp_db.add_segments(
            synth_audio_files["similar2"],
            audio_name=Path(synth_audio_files["similar2"]).name,
            segment_duration_sec=None
        )
        temp_db.add_segments(
            synth_audio_files["different"],
            audio_name=Path(synth_audio_files["different"]).name,
            segment_duration_sec=None
        )

        results = temp_db.search_similar(
            synth_audio_files["similar1"],
            top_k=3,
            duration_sec=None
        )

        expected_keys = {"id", "file", "start_sec", "end_sec", "score"}
        assert set(results[0].keys()) == expected_keys
        assert results[0]["score"] > 0.9

        scores = [r["score"] for r in results]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

        assert "440" in Path(results[0]["file"]).name

    def test_add_segments_prevents_duplicates_file_path_whole_mode(self, temp_db, synth_audio_files):
        # Given: A file path input and expected ID/metadata after first add
        audio_file = synth_audio_files["similar1"]
        expected_id = f"{Path(audio_file).stem}_full"
        expected_count_after_first = 1
        expected_count_after_second = 1

        # When: Add the same file twice (whole mode)
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).stem,
            segment_duration_sec=None
        )
        count_after_first = temp_db.collection.count()
        results_after_first = temp_db.collection.get(ids=[expected_id], include=["metadatas"])

        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).stem,
            segment_duration_sec=None
        )
        count_after_second = temp_db.collection.count()
        results_after_second = temp_db.collection.get(ids=[expected_id], include=["metadatas"])

        # Then: Count doesn't increase, ID exists once, metadata matches expected
        assert count_after_first == expected_count_after_first
        assert count_after_second == expected_count_after_second
        assert len(results_after_first["ids"]) == 1
        assert results_after_first["ids"][0] == expected_id
        assert Path(results_after_first["metadatas"][0]["file"]).resolve() == Path(audio_file).resolve()
        assert len(results_after_second["ids"]) == 1  # Still only one

    def test_add_segments_prevents_duplicates_bytes_whole_mode(self, temp_db, synth_audio_files):
        # Given: Bytes input and expected ID/metadata after first add
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()
        audio_name = "custom_bytes"
        expected_id = f"{audio_name}_full"
        expected_file_meta = audio_name
        expected_count_after_first = 1
        expected_count_after_second = 1

        # When: Add the same bytes twice (whole mode)
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None
        )
        count_after_first = temp_db.collection.count()
        results_after_first = temp_db.collection.get(ids=[expected_id], include=["metadatas"])

        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None
        )
        count_after_second = temp_db.collection.count()
        results_after_second = temp_db.collection.get(ids=[expected_id], include=["metadatas"])

        # Then: Count doesn't increase, ID exists once, metadata matches expected
        assert count_after_first == expected_count_after_first
        assert count_after_second == expected_count_after_second
        assert len(results_after_first["ids"]) == 1
        assert results_after_first["ids"][0] == expected_id
        assert results_after_first["metadatas"][0]["file"] == expected_file_meta
        assert len(results_after_second["ids"]) == 1  # Still only one

    def test_add_segments_prevents_duplicates_bytes_chunked_mode(self, temp_db, synth_audio_files):
        # Given: Bytes input for chunked mode and expected min segments
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()
        audio_name = "chunked_bytes"
        segment_duration_sec = 1.0
        overlap_sec = 0.5
        expected_min_segments = 4   # 2s audio → starts at 0.0, 0.5, 1.0, 1.5 → 4 chunks
        expected_count_after_second = expected_min_segments

        # When: Add the same bytes twice (chunked mode)
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=segment_duration_sec,
            overlap_sec=overlap_sec
        )
        count_after_first = temp_db.collection.count()

        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=segment_duration_sec,
            overlap_sec=overlap_sec
        )
        count_after_second = temp_db.collection.count()

        # Then: Count doesn't increase after re-add
        assert count_after_first >= expected_min_segments
        assert count_after_second == expected_count_after_second

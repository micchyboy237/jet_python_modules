import tempfile
import io
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List
from typing import TypedDict
from chromadb.api.types import QueryResult
from jet.audio.audio_search import (
    AudioSegmentDatabase,
    load_audio_segment,
)

class ExpectedMetadata(TypedDict):
    file: str
    start_sec: float
    end_sec: float

# ------------------------- Fixtures -------------------------

@pytest.fixture
def temp_db() -> AudioSegmentDatabase:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = AudioSegmentDatabase(persist_dir=tmpdir)
        yield db

@pytest.fixture
def synth_audio_files() -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        waveform_440 = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_440 = dir_path / "tone_440.wav"
        torchaudio.save(path_440, waveform_440.unsqueeze(0), 48000)
        waveform_880 = torch.sin(2 * torch.pi * 880 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_880 = dir_path / "tone_880.wav"
        torchaudio.save(path_880, waveform_880.unsqueeze(0), 48000)
        path_440_duplicate = dir_path / "tone_440_duplicate.wav"
        torchaudio.save(path_440_duplicate, waveform_440.unsqueeze(0), 48000)
        yield {
            "similar1": str(path_440),
            "similar2": str(path_440_duplicate),
            "different": str(path_880),
        }

# =================================================================
# 1. Audio Input Normalization & Loading (AudioInputLoader)
# =================================================================

class TestAudioInputLoader:
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
    
    def test_load_audio_segment_handles_ndarray_input(self):
        sr_original = 44100
        duration_sec = 0.8
        t = np.linspace(0, duration_sec, int(sr_original * duration_sec), endpoint=False)
        waveform_np = np.sin(2 * np.pi * 660 * t).astype(np.float32)
        expected_samples = int(48000 * 2.0)
        waveform_tensor = load_audio_segment(
            audio_input=waveform_np,
            start_sec=0.0,
            duration_sec=2.0
        )
        assert isinstance(waveform_tensor, torch.Tensor)
        assert waveform_tensor.ndim == 1
        assert len(waveform_tensor) == expected_samples
        assert not torch.allclose(waveform_tensor[:int(48000 * duration_sec)], torch.zeros_like(waveform_tensor[:int(48000 * duration_sec)]))
        assert torch.allclose(waveform_tensor[int(48000 * duration_sec):], torch.zeros_like(waveform_tensor[int(48000 * duration_sec):]), atol=1e-6)

# =================================================================
# 2. Segment Slicing & Deduplication (AudioSegmenter, SegmentIdFactory)
# =================================================================

class TestAudioSegmenterAndDeduplication:
    def test_add_segments_stores_multiple_segments(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        expected_file_name = Path(audio_file).name
        temp_db.add_segments(
            audio_file,
            audio_name=expected_file_name,
            segment_duration_sec=30.0,
            overlap_sec=0.0
        )
        count = temp_db.collection.count()
        assert count >= 1
        results: QueryResult = temp_db.collection.get(include=["metadatas"])
        metadata = results["metadatas"][0]
        assert Path(metadata["file"]).resolve() == Path(audio_file).resolve()
        assert metadata["start_sec"] == 0.0

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

    def test_add_segments_bytes_whole_file_mode(self, temp_db, synth_audio_files):
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()
        expected_file_meta = "custom_name"
        expected_start_sec = 0.0
        expected_end_sec_approx = 2.0
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="custom_name",
            segment_duration_sec=None,
        )
        result_count = temp_db.collection.count()
        assert result_count == 1
        results = temp_db.collection.get(include=["metadatas", "embeddings"])
        result_id = results["ids"][0]
        assert result_id.startswith("custom_name_")
        assert result_id.endswith("_full")
        result_meta = results["metadatas"][0]
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
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="in_memory_test",
            segment_duration_sec=segment_duration_sec,
            overlap_sec=overlap_sec,
        )
        result_count = temp_db.collection.count()
        assert result_count >= expected_min_segments
        results = temp_db.collection.get(include=["metadatas"])
        ids = results["ids"]
        metadatas = results["metadatas"]
        for suffix in ["_0.0", "_0.5", "_1.0"]:
            assert any(id.endswith(suffix) for id in ids), f"No ID ends with {suffix}"
        start_secs = [m["start_sec"] for m in metadatas]
        assert all(start_secs[i] <= start_secs[i + 1] for i in range(len(start_secs) - 1))
        for meta in metadatas:
            assert meta["file"] == "in_memory_test"

    def test_add_segments_prevents_duplicates_file_path_whole_mode(self, temp_db, synth_audio_files):
        audio_file = synth_audio_files["similar1"]
        expected_count_after_first = 1
        expected_count_after_second = 1
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).stem,
            segment_duration_sec=None
        )
        count_after_first = temp_db.collection.count()
        results_after_first = temp_db.collection.get(include=["metadatas"])
        assert len(results_after_first["ids"]) == 1, "Expected exactly one segment after first add"
        real_id = results_after_first["ids"][0]
        temp_db.add_segments(
            audio_file,
            audio_name=Path(audio_file).stem,
            segment_duration_sec=None
        )
        count_after_second = temp_db.collection.count()
        results_after_second = temp_db.collection.get(ids=[real_id], include=["metadatas"])
        assert count_after_first == expected_count_after_first
        assert count_after_second == expected_count_after_second
        assert len(results_after_first["ids"]) == 1
        assert results_after_first["ids"][0] == real_id
        assert Path(results_after_first["metadatas"][0]["file"]).resolve() == Path(audio_file).resolve()
        assert len(results_after_second["ids"]) == 1

    def test_add_segments_prevents_duplicates_bytes_whole_mode(self, temp_db, synth_audio_files):
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()
        audio_name = "custom_bytes"
        expected_file_meta = audio_name
        expected_count_after_first = 1
        expected_count_after_second = 1
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None
        )
        count_after_first = temp_db.collection.count()
        results_after_first = temp_db.collection.get(include=["metadatas"])
        assert len(results_after_first["ids"]) == 1, "Expected exactly one segment after first add"
        real_id = results_after_first["ids"][0]
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name=audio_name,
            segment_duration_sec=None
        )
        count_after_second = temp_db.collection.count()
        results_after_second = temp_db.collection.get(ids=[real_id], include=["metadatas"])
        assert count_after_first == expected_count_after_first
        assert count_after_second == expected_count_after_second
        assert len(results_after_first["ids"]) == 1
        assert results_after_first["ids"][0] == real_id
        assert results_after_first["metadatas"][0]["file"] == expected_file_meta
        assert len(results_after_second["ids"]) == 1

    def test_add_segments_prevents_duplicates_bytes_chunked_mode(self, temp_db, synth_audio_files):
        reference_file = synth_audio_files["similar1"]
        with open(reference_file, "rb") as f:
            audio_bytes = f.read()
        audio_name = "chunked_bytes"
        segment_duration_sec = 1.0
        overlap_sec = 0.5
        expected_min_segments = 4
        expected_count_after_second = expected_min_segments
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
        assert count_after_first >= expected_min_segments
        assert count_after_second == expected_count_after_second

# =================================================================
# 3. Persistence & Metadata Mapping (SegmentRepository/Store)
#    (Covered in the above integration; no dedicated separation here)
# =================================================================

# =================================================================
# 4. Search Query Preparation (SearchQueryBuilder)
# =================================================================

class TestSearchQueryPreparation:
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

    def test_search_similar_file_and_bytes_queries_are_consistent_self_match(self, temp_db):
        duration_sec = 1.2
        sr = 48000
        t = torch.linspace(0, duration_sec, int(sr * duration_sec))
        waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name
            torchaudio.save(temp_path, waveform, sr)
        try:
            temp_db.add_segments(temp_path, segment_duration_sec=None)
            expected_file = str(Path(temp_path).resolve())
            expected_score_min = 0.99
            results_file = temp_db.search_similar(temp_path, top_k=1)
            result_file = results_file[0]
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            results_bytes = temp_db.search_similar(audio_bytes, top_k=1)
            result_bytes = results_bytes[0]
            assert result_file["file"] == expected_file
            assert result_file["score"] >= expected_score_min
            assert result_bytes["file"] == expected_file
            assert result_bytes["score"] >= expected_score_min
            assert np.isclose(result_file["score"], result_bytes["score"], atol=1e-4)
        finally:
            Path(temp_path).unlink()

    def test_search_similar_bytes_only_workflow_self_match_high_score(self, temp_db):
        duration_sec = 0.8
        sr = 48000
        t = torch.linspace(0, duration_sec, int(sr * duration_sec))
        waveform = torch.sin(2 * torch.pi * 880 * t).unsqueeze(0)
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sr, format="wav")
        audio_bytes = buffer.getvalue()
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="bytes_only_short",
            segment_duration_sec=None,
        )
        expected_score_min = 0.99
        results = temp_db.search_similar(audio_bytes, top_k=1)
        result = results[0]
        assert result["file"] == "bytes_only_short"
        assert result["score"] >= expected_score_min

    def test_add_segments_and_search_with_ndarray_input_self_match(self, temp_db):
        sr = 48000
        duration_sec = 1.5
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        waveform_np = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        temp_db.add_segments(
            audio_input=waveform_np,
            audio_name="ndarray_tone_440",
            segment_duration_sec=None,
        )
        expected_score_min = 0.99
        expected_file_meta = "ndarray_tone_440"
        results = temp_db.search_similar(waveform_np, top_k=1)
        result = results[0]
        assert result["file"] == expected_file_meta
        assert result["score"] >= expected_score_min

# =================================================================
# 5. Similarity Scoring & Ranking (SimilarityRanker)
# =================================================================

class TestSimilarityScoringAndRanking:
    def test_search_finds_bytes_added_segment(self, temp_db, synth_audio_files):
        file_path = synth_audio_files["similar1"]
        duplicate_path = synth_audio_files["similar2"]
        temp_db.add_segments(
            file_path,
            audio_name=Path(file_path).name,
            segment_duration_sec=None
        )
        with open(duplicate_path, "rb") as f:
            audio_bytes = f.read()
        temp_db.add_segments(
            audio_input=audio_bytes,
            audio_name="bytes_duplicate",
            segment_duration_sec=None,
        )
        assert temp_db.collection.count() == 2
        results = temp_db.search_similar(audio_bytes, top_k=2, duration_sec=None)
        assert len(results) == 2
        scores = [r["score"] for r in results]
        assert all(score > 0.85 for score in scores)
        files = {r["file"] for r in results}
        expected_files = {file_path, "bytes_duplicate"}
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

# =================================================================
# 6. Empty / Edge-case Behavior (SearchGuard / SearchResultPolicy)
# =================================================================

class TestSearchGuardAndEdgeCases:
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

# =================================================================
# 7. Query Localization Logic (QueryLocalizer)
# =================================================================

class TestQueryLocalizer:
    def test_search_similar_localize_in_query_returns_query_time_ranges(self, temp_db, synth_audio_files):
        short1_path = synth_audio_files["similar1"]  # 2s 440 Hz tone
        short2_path = synth_audio_files["different"]  # 2s 880 Hz tone
        temp_db.add_segments(short1_path, audio_name="segment_440hz", segment_duration_sec=None)
        temp_db.add_segments(short2_path, audio_name="segment_880hz", segment_duration_sec=None)
        assert temp_db.collection.count() == 2
        sr = 48000
        wave_440, _ = torchaudio.load(short1_path)
        wave_880, _ = torchaudio.load(short2_path)
        if wave_440.shape[0] > 1:
            wave_440 = wave_440.mean(dim=0, keepdim=True)
        if wave_880.shape[0] > 1:
            wave_880 = wave_880.mean(dim=0, keepdim=True)
        silence_5s = torch.zeros(1, int(sr * 5.0))
        query_wave = torch.cat([
            silence_5s,
            wave_440,
            silence_5s,
            wave_880,
            silence_5s
        ], dim=1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
            torchaudio.save(temp_path, query_wave, sr)
        try:
            results = temp_db.search_similar(
                temp_path,
                top_k=3,
                localize_in_query=True
            )
            assert len(results) >= 2
            assert all("query_start_sec" in r for r in results)
            assert all("query_end_sec" in r for r in results)
            match_440 = None
            match_880 = None
            for r in results:
                if "440" in r["file"]:
                    match_440 = r
                elif "880" in r["file"]:
                    match_880 = r
            assert match_440 is not None
            assert match_880 is not None
            assert match_440["query_start_sec"] in {0.0, 5.0}
            assert match_880["query_start_sec"] in {0.0, 5.0}
            assert match_880["query_start_sec"] == 5.0
            assert match_440["score"] > 0.85
            assert match_880["score"] > 0.85
        finally:
            Path(temp_path).unlink()

# =================================================================
# 8. Growing Short-segment Recovery Logic (GrowingQueryMatcher)
# =================================================================

class TestGrowingQueryMatcher:
    def test_search_similar_growing_short_segments_improves_short_query_match(self, temp_db, synth_audio_files):
        tone_440_path = synth_audio_files["similar1"]
        tone_880_path = synth_audio_files["different"]
        temp_db.add_segments(tone_440_path, segment_duration_sec=None)
        temp_db.add_segments(tone_880_path, segment_duration_sec=None)
        assert temp_db.collection.count() == 2
        waveform_440, sr = torchaudio.load(tone_440_path)
        if waveform_440.shape[0] > 1:
            waveform_440 = waveform_440.mean(dim=0, keepdim=True)
        short_samples = int(0.3 * sr)
        short_query = waveform_440[:, :short_samples]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, short_query, sr)
        try:
            normal_results = temp_db.search_similar(short_path, top_k=2)
            normal_top = normal_results[0]
            growing_results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=2
            )
            growing_top = growing_results[0]
            assert "440" in Path(normal_top["file"]).name
            assert "440" in Path(growing_top["file"]).name
            assert growing_top["score"] > normal_top["score"] + 0.1
            assert growing_top["score"] > 0.85
            assert "query_start_sec" not in growing_top
            assert "query_end_sec" not in growing_top
        finally:
            Path(short_path).unlink()

    def test_search_similar_growing_short_segments_handles_very_short_query(self, temp_db, synth_audio_files):
        tone_440_path = synth_audio_files["similar1"]
        temp_db.add_segments(tone_440_path, segment_duration_sec=None)
        assert temp_db.collection.count() == 1
        waveform, sr = torchaudio.load(tone_440_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        very_short_samples = int(0.05 * sr)
        very_short = waveform[:, :very_short_samples]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, very_short, sr)
        try:
            normal_results = temp_db.search_similar(short_path, top_k=1)
            growing_results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=1
            )
            normal_score = normal_results[0]["score"]
            growing_score = growing_results[0]["score"]
            assert "440" in Path(normal_results[0]["file"]).name
            assert "440" in Path(growing_results[0]["file"]).name
            assert normal_score > 0.75
            assert growing_score > 0.75
            assert growing_score >= normal_score - 1e-4
        finally:
            Path(short_path).unlink()

    def test_growing_short_segments_attaches_prefix_fields(self, temp_db, synth_audio_files):
        """Given: Database with two distinct tones
        When: Searching short 440 Hz prefix with growing mode
        Then: Results contain 'prefix_scores' and 'prefix_durations_sec' for each match
        """
        # Index full tones
        temp_db.add_segments(synth_audio_files["similar1"], segment_duration_sec=None)
        temp_db.add_segments(synth_audio_files["different"], segment_duration_sec=None)

        # Create 0.4s short query from 440 Hz tone
        waveform, sr = torchaudio.load(synth_audio_files["similar1"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        short_samples = int(0.4 * sr)
        short_query = waveform[:, :short_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, short_query, sr)

        try:
            results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=2
            )

            # Expected: 4 prefixes → 0.1s, 0.2s, 0.3s, 0.4s
            expected_prefix_count = 4
            expected_durations = [0.1, 0.2, 0.3, 0.4]

            assert len(results) == 2
            for result in results:
                assert "prefix_scores" in result
                assert "prefix_durations_sec" in result
                assert len(result["prefix_scores"]) == expected_prefix_count
                assert result["prefix_durations_sec"] == expected_durations

            # Top match should be the 440 Hz tone and its final score equals max of its prefix scores
            top_match = results[0]
            assert "440" in Path(top_match["file"]).name
            assert top_match["score"] == max(top_match["prefix_scores"])

        finally:
            Path(short_path).unlink()

    def test_growing_prefix_scores_increase_for_matching_segment(self, temp_db, synth_audio_files):
        """Given: Short prefix of 440 Hz tone
        When: Growing search
        Then: Prefix scores for the matching 440 Hz segment generally increase with longer prefixes
        """
        temp_db.add_segments(synth_audio_files["similar1"], segment_duration_sec=None)
        temp_db.add_segments(synth_audio_files["different"], segment_duration_sec=None)

        waveform, sr = torchaudio.load(synth_audio_files["similar1"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        short_samples = int(0.4 * sr)
        short_query = waveform[:, :short_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, short_query, sr)

        try:
            results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=1
            )

            match_440 = results[0]
            scores = match_440["prefix_scores"]

            # Overall trend: longer prefix → better score
            assert scores[-1] > scores[0]  # Final 0.4s > initial 0.1s
            assert match_440["score"] == max(scores)

        finally:
            Path(short_path).unlink()

    def test_growing_mode_fallback_for_too_short_query(self, temp_db, synth_audio_files):
        """Given: Database with tone
        When: Querying with <0.1s audio
        Then: Falls back to normal search → no prefix fields attached
        """
        temp_db.add_segments(synth_audio_files["similar1"], segment_duration_sec=None)

        waveform, sr = torchaudio.load(synth_audio_files["similar1"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        very_short_samples = int(0.05 * sr)  # 50 ms
        very_short = waveform[:, :very_short_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, very_short, sr)

        try:
            results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=1
            )

            assert len(results) == 1
            assert "prefix_scores" not in results[0]
            assert "prefix_durations_sec" not in results[0]

        finally:
            Path(short_path).unlink()

    def test_growing_prefix_scores_are_independent_per_db_segment(self, temp_db, synth_audio_files):
        """Given: Multiple DB segments
        When: Growing search
        Then: Each result has its own independent prefix score list
        """
        temp_db.add_segments(synth_audio_files["similar1"], segment_duration_sec=None)
        temp_db.add_segments(synth_audio_files["different"], segment_duration_sec=None)

        waveform, sr = torchaudio.load(synth_audio_files["similar1"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        short_samples = int(0.4 * sr)
        short_query = waveform[:, :short_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            short_path = f.name
            torchaudio.save(short_path, short_query, sr)

        try:
            results = temp_db.search_similar(
                short_path,
                use_growing_short_segments=True,
                top_k=2
            )

            assert len(results) == 2
            assert len(results[0]["prefix_scores"]) == len(results[1]["prefix_scores"])
            assert results[0]["prefix_scores"] != results[1]["prefix_scores"]

        finally:
            Path(short_path).unlink()

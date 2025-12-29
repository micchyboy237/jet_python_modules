import tempfile
import io  # Added for use with BytesIO
from pathlib import Path
from typing import List

import pytest
import torch
import torchaudio
from chromadb.api.types import QueryResult

from jet.audio.audio_search import (  # Replace with actual module name, e.g., audio_search
    AudioSegmentDatabase,
    load_audio_segment,
)

# Fixture: Create a temporary database and clean up afterwards
@pytest.fixture
def temp_db() -> AudioSegmentDatabase:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = AudioSegmentDatabase(persist_dir=tmpdir)
        yield db
    # TemporaryDirectory cleans up automatically

# Fixture: Generate simple synthetic audio segments (sine waves at different frequencies)
@pytest.fixture
def synth_audio_files() -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        
        # 440 Hz tone (A4) – represents one "sound"
        waveform_440 = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_440 = dir_path / "tone_440.wav"
        torchaudio.save(path_440, waveform_440.unsqueeze(0), 48000)
        
        # 880 Hz tone (A5) – clearly different
        waveform_880 = torch.sin(2 * torch.pi * 880 * torch.linspace(0, 2.0, steps=int(48000 * 2.0)))
        path_880 = dir_path / "tone_880.wav"
        torchaudio.save(path_880, waveform_880.unsqueeze(0), 48000)
        
        # Another 440 Hz tone – should be very similar to first
        path_440_duplicate = dir_path / "tone_440_duplicate.wav"
        torchaudio.save(path_440_duplicate, waveform_440.unsqueeze(0), 48000)
        
        yield {
            "similar1": str(path_440),
            "similar2": str(path_440_duplicate),
            "different": str(path_880),
        }

class TestLoadAudioSegment:
    def test_loads_file_and_returns_correct_shape(self, synth_audio_files):
        # Given
        audio_path = synth_audio_files["similar1"]
        expected_samples = int(48000 * 30.0)  # default duration
        
        # When
        waveform = load_audio_segment(audio_path, start_sec=0.0, duration_sec=30.0)
        
        # Then
        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 1
        assert len(waveform) == expected_samples
        assert waveform.dtype == torch.float32

    def test_handles_raw_bytes_input(self, synth_audio_files):
        # Given
        audio_path = synth_audio_files["similar1"]
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        expected_samples = int(48000 * 10.0)
        
        # When
        waveform = load_audio_segment(audio_bytes, duration_sec=10.0)
        
        # Then
        assert isinstance(waveform, torch.Tensor)
        assert len(waveform) == expected_samples

    def test_pads_short_segment(self):
        # Given a very short audio (0.5s sine)
        short_wave = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 0.5, steps=int(48000 * 0.5)))
        short_bytes_io = io.BytesIO()
        torchaudio.save(short_bytes_io, short_wave.unsqueeze(0), 48000, format="wav")
        short_bytes = short_bytes_io.getvalue()
        expected_samples = int(48000 * 30.0)
        
        # When
        waveform = load_audio_segment(short_bytes, duration_sec=30.0)
        
        # Then
        assert len(waveform) == expected_samples
        # Padding should be zeros
        assert torch.all(waveform[int(48000 * 0.5):] == 0)

class TestAudioSegmentDatabase:
    def test_add_segments_from_file_stores_multiple_segments(self, temp_db, synth_audio_files):
        # Given
        audio_file = synth_audio_files["similar1"]
        
        # When
        temp_db.add_segments_from_file(audio_file, segment_duration_sec=30.0, overlap_sec=0.0)
        
        # Then
        count = temp_db.collection.count()
        # 2-second file with 30s segments → only 1 segment (padded)
        assert count >= 1
        
        results: QueryResult = temp_db.collection.get(include=["metadatas"])
        metadata = results["metadatas"][0]
        assert metadata["file"] == audio_file
        assert metadata["start_sec"] == 0.0

    def test_search_returns_self_as_most_similar(self, temp_db, synth_audio_files):
        # Given three short audio segments: two identical 440Hz tones and one different 880Hz tone
        file1 = synth_audio_files["similar1"]      # 440Hz, 2 seconds
        file2 = synth_audio_files["similar2"]      # identical 440Hz duplicate
        file_diff = synth_audio_files["different"]  # 880Hz
        
        temp_db.add_segments_from_file(file1, segment_duration_sec=30.0, overlap_sec=0.0)
        temp_db.add_segments_from_file(file2, segment_duration_sec=30.0, overlap_sec=0.0)
        temp_db.add_segments_from_file(file_diff, segment_duration_sec=30.0, overlap_sec=0.0)
        
        # When querying using the first 440Hz file as the query audio
        results: List[dict] = temp_db.search_similar(file1, top_k=3)
        
        # Then the top result should be a 440Hz tone
        top_result = results[0]
        top_file_name = Path(top_result["file"]).name
        top_tone = "440" if "440" in top_file_name else "880"
        expected_top_tone = "440"
        
        assert top_tone == expected_top_tone
        
        # Expect very high similarity for top match (accounting for padding effects in short clips)
        max_allowed_distance = 0.30
        expected_low_distance = top_result["distance"] < max_allowed_distance
        
        assert expected_low_distance
        
        # Overall ranking checks
        actual_contains = ["440" if "440" in Path(r["file"]).name else "880" for r in results]
        expected_at_least_two_similar = actual_contains.count("440") >= 2
        expected_different_ranks_lower = actual_contains[-1] == "880"
        expected_non_decreasing_distances = results[0]["distance"] <= results[1]["distance"] <= results[2]["distance"]
        
        assert expected_at_least_two_similar
        assert expected_different_ranks_lower
        assert expected_non_decreasing_distances

    def test_search_with_bytes_query_works(self, temp_db, synth_audio_files):
        # Given database populated with one 440Hz and one 880Hz segment
        temp_db.add_segments_from_file(synth_audio_files["similar1"])
        temp_db.add_segments_from_file(synth_audio_files["different"])
        
        # When querying with raw bytes from the duplicate 440Hz tone
        with open(synth_audio_files["similar2"], "rb") as f:
            query_bytes = f.read()
        results: List[dict] = temp_db.search_similar(query_bytes, top_k=2)
        
        # Then the top result should be the 440Hz tone
        top_result = results[0]
        top_file_name = Path(top_result["file"]).name
        top_tone = "440" if "440" in top_file_name else "880"
        expected_top_tone = "440"
        
        assert top_tone == expected_top_tone
        
        max_allowed_distance = 0.30
        expected_low_distance = top_result["distance"] < max_allowed_distance
        
        assert expected_low_distance
        
        actual_contains = ["440" if "440" in Path(r["file"]).name else "880" for r in results]
        expected_similar_first = actual_contains[0] == "440"
        expected_different_last = actual_contains[-1] == "880"
        expected_non_decreasing_distances = results[0]["distance"] <= results[1]["distance"]
        
        assert expected_similar_first
        assert expected_different_last
        assert expected_non_decreasing_distances

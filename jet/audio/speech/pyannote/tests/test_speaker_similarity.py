import numpy as np
import pytest
from pyannote.core import Segment
from jet.audio.speech.pyannote.speaker_similarity import SpeakerEmbedding


# ----------------------------
# Mocked inference for testing
# ----------------------------
class DummyInference:
    """
    Simulates pyannote.audio.Inference behavior.
    This lets tests run without downloading any models.
    """

    def __call__(self, file_path: str):
        # Deterministic 1D vectors based on file names
        if "speaker1" in file_path:
            return np.array([0.1, 0.2, 0.3])
        if "speaker2" in file_path:
            return np.array([0.9, 0.1, 0.4])
        return np.array([1.0, 1.0, 1.0])

    def crop(self, file_path: str, segment: Segment):
        # Deterministic output based on segment duration
        dur = segment.end - segment.start
        return np.array([dur, dur + 1, dur + 2])


# ----------------------------
# Tests for whole-window inference
# ----------------------------
class TestSpeakerEmbeddingWhole:

    @pytest.fixture
    def tool(self, monkeypatch):
        from pyannote.audio import Model
        monkeypatch.setattr(Model, "from_pretrained", lambda *a, **kw: None)
        spk = SpeakerEmbedding(model_id="dummy")
        monkeypatch.setattr(spk, "_get_inference", lambda: DummyInference())
        return spk

    def test_embed_whole_file(self, tool):
        result = tool.embed("speaker1.wav")
        expected = np.array([[0.1, 0.2, 0.3]])
        assert isinstance(result, dict)
        assert np.allclose(result["vector"], expected)

    def test_distance_whole_files(self, tool):
        dist = tool.distance("speaker1.wav", "speaker2.wav")
        v1 = np.array([[0.1, 0.2, 0.3]])
        v2 = np.array([[0.9, 0.1, 0.4]])
        expected = float(1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert dist == expected

    def test_similarity_whole_files(self, tool):
        sim = tool.similarity("speaker1.wav", "speaker2.wav")
        v1 = np.array([[0.1, 0.2, 0.3]])
        v2 = np.array([[0.9, 0.1, 0.4]])
        expected_sim = 1.0 - float(1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert sim == expected_sim
        assert -1.0 <= sim <= 1.0


# ----------------------------
# Tests for sliding-window inference
# ----------------------------
class TestSpeakerEmbeddingSliding:

    @pytest.fixture
    def tool(self, monkeypatch):
        from pyannote.audio import Model
        monkeypatch.setattr(Model, "from_pretrained", lambda *a, **kw: None)
        spk = SpeakerEmbedding(model_id="dummy")
        # mock sliding inference
        monkeypatch.setattr(spk, "_get_inference", lambda: DummyInference())
        return spk

    def test_embed_segment_sliding(self, tool):
        result = tool.embed("any.wav", start=5.0, end=8.0)
        expected = np.array([[3.0, 4.0, 5.0]])
        assert np.allclose(result["vector"], expected)

    def test_distance_segments_sliding(self, tool):
        dist = tool.distance(
            "any.wav", "any.wav", start1=2.0, end1=5.0, start2=1.0, end2=2.0
        )
        v1 = np.array([[3.0, 4.0, 5.0]])
        v2 = np.array([[1.0, 2.0, 3.0]])
        expected = float(1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert dist == expected

    def test_similarity_segments_sliding(self, tool):
        sim = tool.similarity(
            "any.wav", "any.wav", start1=2.0, end1=5.0, start2=1.0, end2=2.0
        )
        v1 = np.array([[3.0, 4.0, 5.0]])
        v2 = np.array([[1.0, 2.0, 3.0]])
        expected_sim = 1.0 - float(1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert sim == expected_sim
        assert -1.0 <= sim <= 1.0

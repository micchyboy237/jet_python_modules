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
# Fixtures
# ----------------------------
@pytest.fixture
def tool(monkeypatch):
    """
    Provide SpeakerEmbedding instance with all pyannote internals mocked.
    Prevents HF model downloads.
    """

    # --- NEW: mock model load so HF is never queried ---
    from pyannote.audio import Model
    monkeypatch.setattr(Model, "from_pretrained", lambda *args, **kwargs: None)

    spk = SpeakerEmbedding(model_id="dummy", token=None)

    # Mock inference creation
    monkeypatch.setattr(spk, "_get_inference", lambda: DummyInference())

    return spk


# ----------------------------
# Tests
# ----------------------------
class TestSpeakerEmbeddingAPI:

    def test_embed_whole_file(self, tool):
        """
        Given a simple audio path
        When calling embed()
        Then the vector should be reshaped to (1, D) and match deterministic values
        """
        result = tool.embed("speaker1.wav")

        expected = np.array([[0.1, 0.2, 0.3]])
        assert isinstance(result, dict)
        assert np.allclose(result["vector"], expected)

    def test_embed_segment(self, tool):
        """
        Given start and end timestamps
        When calling embed() with segment arguments
        Then crop() output should reflect deterministic duration logic
        """
        result = tool.embed("any.wav", start=10.0, end=13.0)

        # duration = 3.0 → mock returns [3,4,5]
        expected = np.array([[3.0, 4.0, 5.0]])
        assert np.allclose(result["vector"], expected)

    def test_distance_whole_files(self, tool):
        """
        Given two audio files
        When calling distance()
        Then cosine distance must be computed correctly
        """
        dist = tool.distance("speaker1.wav", "speaker2.wav")

        v1 = np.array([[0.1, 0.2, 0.3]])
        v2 = np.array([[0.9, 0.1, 0.4]])
        expected = float(
            (1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        )

        assert dist == expected

    def test_distance_segments(self, tool):
        """
        Given two segments in two files
        When computing distance()
        Then values must match deterministic mock logic
        """
        # segment1 duration = 2 → vec1 = [2,3,4]
        # segment2 duration = 1 → vec2 = [1,2,3]
        dist = tool.distance(
            "any.wav",
            "any.wav",
            start1=10.0, end1=12.0,
            start2=5.0, end2=6.0,
        )

        v1 = np.array([[2.0, 3.0, 4.0]])
        v2 = np.array([[1.0, 2.0, 3.0]])
        expected = float(
            (1 - np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        )

        assert dist == expected

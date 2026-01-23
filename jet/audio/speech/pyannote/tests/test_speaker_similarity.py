# tests/test_speaker_similarity.py
import pytest
from typing import List

from jet.audio.speech.pyannote.speaker_similarity import SpeakerSimilarity


@pytest.fixture
def speaker_sim():
    """Fixture providing SpeakerSimilarity instance (device irrelevant for mocked tests)."""
    return SpeakerSimilarity(hf_token="dummy-token-for-testing", device="cpu")


def fake_get_embedding(self, input_data):
    """Deterministic fake embedding based on input identity (e.g. filename or object id)."""
    return [0.0] * 512  # dummy vector, normalized later in real code but irrelevant


@pytest.mark.parametrize(
    "case_name, inputs, expected_labels, threshold",
    [
        ("two_same_speaker",          ["speakerA_clip1.wav", "speakerA_clip2.wav"], [0, 0],   0.80),
        ("two_different_speakers",    ["speakerA.wav",       "speakerB.wav"],       [0, 1],   0.80),
        ("three_mixed_two_same",      ["A1.wav", "A2.wav", "B.wav"],               [0, 0, 1], 0.80),
        ("three_identical",           ["A1.wav", "A2.wav", "A3.wav"],              [0, 0, 0], 0.80),
        ("three_all_different",       ["S1.wav", "S2.wav", "S3.wav"],              [0, 1, 2], 0.80),
        ("borderline_below_threshold",["A.wav", "A_borderline.wav"],               [0, 1],   0.82),
    ],
    ids=lambda x: x[0] if isinstance(x, tuple) else None
)
def test_assign_speaker_labels_various_scenarios(
    speaker_sim: SpeakerSimilarity,
    monkeypatch,
    case_name: str,
    inputs: List[str],
    expected_labels: List[int],
    threshold: float,
):
    """
    Given: a list of audio inputs and a similarity threshold
    When:  assign_speaker_labels is called
    Then:  returns expected cluster labels using greedy assignment
    """

    def fake_similarity(self, inp1, inp2) -> float:
        idx1 = inputs.index(inp1)
        idx2 = inputs.index(inp2)
        if case_name in ("two_same_speaker", "three_identical"):
            return 0.95
        elif case_name == "two_different_speakers":
            return 0.30
        elif case_name == "three_all_different":
            return 0.30
        elif case_name == "three_mixed_two_same":
            if {idx1, idx2} == {0, 1}:
                return 0.92
            else:
                return 0.25
        elif case_name == "borderline_below_threshold":
            return 0.81  # < 0.82 → should start new cluster
        else:
            raise ValueError(f"Missing fake_similarity definition for case: {case_name}")

    monkeypatch.setattr(SpeakerSimilarity, "similarity",   fake_similarity)
    monkeypatch.setattr(SpeakerSimilarity, "get_embedding", fake_get_embedding)

    # When
    result_labels, result_embs = speaker_sim.assign_speaker_labels(
        inputs=inputs,
        threshold=threshold
    )

    # Then
    expected = expected_labels
    result   = result_labels
    assert result == expected, \
        f"[{case_name}] Labels mismatch: expected {expected}, got {result}"
    assert len(result_embs) == len(inputs)


def test_assign_speaker_labels_raises_on_less_than_two_inputs(
    speaker_sim: SpeakerSimilarity,
):
    """
    Given: fewer than 2 inputs
    When: assign_speaker_labels is called
    Then: raises ValueError with correct message
    """
    # Given
    bad_inputs = ["only_one.wav"]

    # When / Then
    with pytest.raises(ValueError) as exc_info:
        speaker_sim.assign_speaker_labels(bad_inputs, threshold=0.80)

    expected_msg_snippet = "at least 2 inputs"
    result_msg = str(exc_info.value)
    assert expected_msg_snippet in result_msg, f"Expected '{expected_msg_snippet}' in error, got '{result_msg}'"


def test_assign_speaker_labels_returns_embeddings(
    speaker_sim: SpeakerSimilarity,
    monkeypatch,
):
    """
    Given: valid inputs
    When: assign_speaker_labels is called
    Then: second return value is list of embeddings with correct length
    """
    # Given
    inputs = ["file1.wav", "file2.wav"]

    def fake_similarity(self, a, b):
        return 0.9

    monkeypatch.setattr(SpeakerSimilarity, "similarity", fake_similarity)
    monkeypatch.setattr(SpeakerSimilarity, "get_embedding", fake_get_embedding)

    # When
    labels, embeddings = speaker_sim.assign_speaker_labels(inputs, threshold=0.5)

    # Then
    expected_len = len(inputs)
    result_len = len(embeddings)
    assert result_len == expected_len
    assert all(isinstance(e, list) for e in embeddings)  # dummy type check
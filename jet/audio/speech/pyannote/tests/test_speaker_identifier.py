# tests/test_speaker_identifier.py
import pytest
import numpy as np
from unittest.mock import Mock

from jet.audio.speech.pyannote.speaker_identifier import SpeakerIdentifier

# Fake 512-dim unit vector factory
def fake_unit_embedding(dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def identifier(mocker):
    """Fresh instance with mocked heavy parts."""
    # Mock the model loading and inference
    mock_model = Mock()
    mocker.patch("pyannote.audio.Model.from_pretrained", return_value=mock_model)

    mock_inference = Mock()
    mocker.patch("pyannote.audio.Inference", return_value=mock_inference)

    ident = SpeakerIdentifier(
        model_name="pyannote/embedding",
        similarity_threshold=0.78,
        unknown_threshold=0.68,
        min_duration=2.0
    )
    ident.model = mock_model
    ident.inference = mock_inference
    return ident, mock_inference


def test_given_no_references_when_identify_then_unknown(identifier):
    # Given
    ident, _ = identifier

    # When
    label, sim = ident.identify("fake/test.wav")

    # Then
    assert label == "UNKNOWN"
    assert sim == 0.0


def test_given_one_reference_when_identify_same_speaker_then_match(identifier):
    # Given
    ident, mock_inf = identifier
    ref_emb = fake_unit_embedding(seed=100)
    test_emb = fake_unit_embedding(seed=100) + 0.02  # very close
    test_emb /= np.linalg.norm(test_emb)

    # Mock reference addition
    mock_inf.side_effect = lambda p: np.array([[ref_emb]])  # shape (1, 512)
    ident.add_reference("Alice", ["ref_alice.wav"])

    # Mock test audio
    mock_inf.side_effect = lambda p: np.array([[test_emb]]) if "test" in str(p) else np.array([[ref_emb]])

    # When
    label, sim = ident.identify("test_alice.wav")

    # Then
    assert label == "Alice"
    assert sim >= 0.78
    assert sim <= 1.0


def test_given_multiple_references_when_identify_closest_then_correct_label(identifier):
    # Given
    ident, mock_inf = identifier

    alice_emb = fake_unit_embedding(seed=10)
    bob_emb   = fake_unit_embedding(seed=20)
    test_emb  = fake_unit_embedding(seed=12)  # closer to Alice

    mock_inf.side_effect = [
        np.array([[alice_emb]] * 2),  # two refs for Alice → average
        np.array([[bob_emb]]),
        np.array([[test_emb]])        # test
    ]

    ident.add_reference("Alice", ["a1.wav", "a2.wav"])
    ident.add_reference("Bob", ["b1.wav"])

    # When
    label, sim = ident.identify("test_close_to_alice.wav")

    # Then
    assert label == "Alice"
    assert sim > ident.similarity_threshold


def test_given_medium_similarity_when_identify_then_uncertain(identifier):
    # Given
    ident, mock_inf = identifier
    ref_emb = fake_unit_embedding(seed=50)
    test_emb = fake_unit_embedding(seed=55)  # medium distance ~0.72 sim

    mock_inf.side_effect = [
        np.array([[ref_emb]]),
        np.array([[test_emb]])
    ]

    ident.add_reference("Charlie", ["ref.wav"])

    # When
    label, sim = ident.identify("medium.wav")

    # Then
    assert "UNCERTAIN_Charlie" in label
    assert ident.unknown_threshold <= sim < ident.similarity_threshold


def test_given_low_similarity_when_identify_then_unknown(identifier):
    # Given
    ident, mock_inf = identifier
    ref_emb = fake_unit_embedding(seed=30)
    test_emb = fake_unit_embedding(seed=80)  # low sim ~0.4

    mock_inf.side_effect = [
        np.array([[ref_emb]]),
        np.array([[test_emb]])
    ]

    ident.add_reference("Dana", ["ref_dana.wav"])

    # When
    label, sim = ident.identify("stranger.wav")

    # Then
    assert label == "UNKNOWN"
    assert sim < ident.unknown_threshold


def test_given_existing_speaker_when_add_without_force_then_no_overwrite(identifier):
    # Given
    ident, mock_inf = identifier
    old_emb = fake_unit_embedding(seed=200)
    new_emb = fake_unit_embedding(seed=300)

    mock_inf.side_effect = [
        np.array([[old_emb]]),
        np.array([[new_emb]])
    ]

    ident.add_reference("Eve", ["old_eve.wav"])

    # When
    success = ident.add_reference("Eve", ["new_eve.wav"])  # no force

    # Then
    assert success is False
    assert np.allclose(ident.references["Eve"], old_emb)


def test_given_force_overwrite_when_add_then_updates_reference(identifier):
    # Given
    ident, mock_inf = identifier
    old_emb = fake_unit_embedding(seed=200)
    new_emb = fake_unit_embedding(seed=300)

    mock_inf.side_effect = [
        np.array([[old_emb]]),
        np.array([[new_emb]])
    ]

    ident.add_reference("Frank", ["old.wav"])

    # When
    success = ident.add_reference("Frank", ["new.wav"], force_overwrite=True)

    # Then
    assert success is True
    assert np.allclose(ident.references["Frank"], new_emb)


def test_given_multiple_ref_files_when_add_then_averages_correctly(identifier):
    # Given
    ident, mock_inf = identifier

    emb1 = fake_unit_embedding(seed=400)
    emb2 = fake_unit_embedding(seed=401)  # very similar

    mock_inf.side_effect = [
        np.array([[emb1]]),
        np.array([[emb2]])
    ]

    # When
    ident.add_reference("Grace", ["g1.wav", "g2.wav"])

    # Then
    stored = ident.references["Grace"]
    expected_avg = (emb1 + emb2) / 2
    expected_avg /= np.linalg.norm(expected_avg)
    assert np.allclose(stored, expected_avg, atol=1e-6)
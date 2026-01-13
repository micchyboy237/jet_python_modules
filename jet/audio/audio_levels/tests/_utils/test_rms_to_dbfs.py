# tests/test_rms_to_dbfs.py
import pytest

from jet.audio.audio_levels.utils import rms_to_dbfs


@pytest.mark.parametrize(
    "rms, expected_db",
    [
        (1.0, 0.0),
        (0.7071067811865475, -3.0),
        (0.5, -6.020599913279624),
        (0.316227766, -10.0),
        (0.1, -20.0),
        (0.0316227766, -30.0),
        (0.0, float("-inf")),
        (1e-8, pytest.approx(-160.0, abs=1.0)),
    ],
    ids=["0dBFS", "-3dB", "-6dB", "-10dB", "-20dB", "-30dB", "silence", "very_very_quiet"]
)
def test_rms_to_dbfs_standard_values(rms, expected_db):
    """Given linear RMS value, when converting to dBFS, then result is correct"""
    result = rms_to_dbfs(rms)

    if expected_db == float("-inf"):
        assert result == float("-inf")
    else:
        assert result == pytest.approx(expected_db, abs=0.01)


def test_rms_to_dbfs_negative_input():
    """Given negative RMS value, should raise appropriate error"""
    with pytest.raises(ValueError, match="RMS.*negative"):
        rms_to_dbfs(-0.01)

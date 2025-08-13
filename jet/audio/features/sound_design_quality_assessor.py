import logging
from typing import Dict, Optional
from jet.audio.audio_file_analyzer import AudioFileAnalyzer
from jet.logger import logger


def assess_audio_quality(file_path: str) -> Dict[str, any]:
    """Assess audio quality for sound design based on RMS energy and spectral centroid.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary with quality assessment results.
    """
    logger.setLevel(logging.INFO)
    analyzer = AudioFileAnalyzer(file_path)
    features = analyzer.get_audio_features()

    result = {
        "file_path": file_path,
        "is_acceptable": False,
        "issues": []
    }

    if not features:
        result["issues"].append("Failed to extract audio features")
        return result

    # Quality thresholds
    min_rms = 0.1  # Minimum loudness
    max_rms = 0.7  # Maximum loudness to avoid clipping
    min_centroid = 200.0  # Minimum tonal balance
    max_centroid = 4000.0  # Maximum tonal balance

    rms = features.get("rms_energy", 0.0)
    centroid = features.get("spectral_centroid_hz", 0.0)

    if rms < min_rms:
        result["issues"].append(
            f"RMS energy {rms:.3f} too low, minimum is {min_rms}")
    elif rms > max_rms:
        result["issues"].append(
            f"RMS energy {rms:.3f} too high, maximum is {max_rms}")

    if centroid < min_centroid:
        result["issues"].append(
            f"Spectral centroid {centroid:.1f} Hz too low, minimum is {min_centroid}")
    elif centroid > max_centroid:
        result["issues"].append(
            f"Spectral centroid {centroid:.1f} Hz too high, maximum is {max_centroid}")

    result["is_acceptable"] = len(result["issues"]) == 0
    result["features"] = features

    if result["is_acceptable"]:
        logger.info(f"Audio file {file_path} meets quality standards")
    else:
        logger.warning(
            f"Audio file {file_path} failed quality assessment: {result['issues']}")

    return result


if __name__ == "__main__":
    audio_file = "sample_audio.wav"  # Replace with real audio file path
    quality_result = assess_audio_quality(audio_file)
    print(f"Quality Assessment: {quality_result}")

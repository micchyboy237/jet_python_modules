import json
from typing import Dict, Optional
from jet.audio.audio_file_analyzer import AudioFileAnalyzer
from jet.logger import logger


def extract_features_for_ml(file_path: str, output_json: Optional[str] = None) -> Dict[str, any]:
    """Extract audio features for music genre classification.

    Args:
        file_path: Path to the audio file.
        output_json: Optional path to save features as JSON.

    Returns:
        Dictionary with extracted features.
    """
    logger.setLevel(logging.INFO)
    analyzer = AudioFileAnalyzer(file_path)
    features = analyzer.get_audio_features()

    if not features:
        logger.error(f"No features extracted for {file_path}")
        return {}

    # Normalize features for ML (e.g., scale to [0, 1] where applicable)
    normalized_features = {
        # Normalize pitch (0-1000 Hz range)
        "mean_pitch_hz": features.get("mean_pitch_hz", 0.0) / 1000.0,
        # Normalize tempo (0-200 BPM range)
        "tempo_bpm": features.get("tempo_bpm", 0.0) / 200.0,
        # Normalize centroid
        "spectral_centroid_hz": features.get("spectral_centroid_hz", 0.0) / 8000.0,
        # Cap RMS at 1.0
        "rms_energy": min(features.get("rms_energy", 0.0), 1.0)
    }

    logger.info(
        f"Extracted and normalized features for {file_path}: {normalized_features}")

    if output_json:
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(normalized_features, f, indent=2)
            logger.info(f"Saved features to {output_json}")
        except Exception as e:
            logger.error(f"Failed to save features to {output_json}: {str(e)}")

    return normalized_features


if __name__ == "__main__":
    audio_file = "sample_audio.wav"  # Replace with real audio file path
    output_file = "features.json"
    features = extract_features_for_ml(audio_file, output_file)
    print(f"Extracted Features: {features}")

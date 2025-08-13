import logging
from typing import Dict, Optional
from jet.audio.audio_file_analyzer import AudioFileAnalyzer
from jet.logger import logger


def validate_podcast_audio(file_path: str) -> Dict[str, any]:
    """Validate an audio file for podcast processing.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary with validation status and details.
    """
    logger.setLevel(logging.INFO)
    analyzer = AudioFileAnalyzer(file_path)
    metadata = analyzer.get_basic_metadata()

    result = {
        "file_path": file_path,
        "is_valid": False,
        "issues": []
    }

    if not metadata:
        result["issues"].append("Failed to extract metadata")
        return result

    # Podcast requirements
    if metadata.get("file_format") not in ["WAV", "MP3"]:
        result["issues"].append(
            f"Invalid format: {metadata.get('file_format')}, expected WAV or MP3")
    if metadata.get("sample_rate") < 44100:
        result["issues"].append(
            f"Sample rate {metadata.get('sample_rate')} Hz is below 44100 Hz")
    if metadata.get("channels") not in [1, 2]:
        result["issues"].append(
            f"Channels {metadata.get('channels')} not mono or stereo")
    if metadata.get("duration_s") < 30.0:
        result["issues"].append(
            f"Duration {metadata.get('duration_s')}s is less than 30s")

    result["is_valid"] = len(result["issues"]) == 0
    if result["is_valid"]:
        logger.info(f"Audio file {file_path} is valid for podcast processing")
    else:
        logger.warning(
            f"Audio file {file_path} failed validation: {result['issues']}")

    result["metadata"] = metadata
    return result


if __name__ == "__main__":
    audio_file = "sample_audio.wav"  # Replace with real audio file path
    validation_result = validate_podcast_audio(audio_file)
    print(f"Validation Result: {validation_result}")

import os
import time
from typing import Generator, Tuple
from faster_whisper import WhisperModel
from jet.logger import logger


def initialize_whisper_model(model_size: str = "small", device: str = "auto", compute_type: str = "float16") -> WhisperModel:
    """Initialize the Faster-Whisper model."""
    try:
        model = WhisperModel(model_size, device=device,
                             compute_type=compute_type)
        logger.info(
            f"Initialized Whisper model '{model_size}' on {device} with {compute_type} precision")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Whisper model: {e}")
        raise


def transcribe_audio_stream(
    audio_file: str,
    model: WhisperModel,
    language: str = "en",
    vad_filter: bool = True,
    vad_parameters: dict = None
) -> Generator[Tuple[float, float, str], None, None]:
    """Transcribe audio file in near real-time using Faster-Whisper."""
    try:
        # Default VAD parameters if not provided
        if vad_parameters is None:
            vad_parameters = {"min_silence_duration_ms": 500}

        # Wait for initial audio data
        start_time = time.time()
        # Minimum WAV header size
        while not os.path.exists(audio_file) or os.path.getsize(audio_file) < 44:
            if time.time() - start_time > 5:
                logger.error(
                    f"Audio file {audio_file} not found or empty after 5 seconds")
                return
            time.sleep(0.1)

        logger.info(f"Starting transcription for {audio_file}")
        segments, info = model.transcribe(
            audio_file,
            language=language,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters
        )

        logger.info(
            f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        for segment in segments:
            yield segment.start, segment.end, segment.text
            logger.debug(
                f"Transcribed segment [{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

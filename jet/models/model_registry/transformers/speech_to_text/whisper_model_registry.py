from typing import Literal, Optional, Union, List
from faster_whisper import WhisperModel

from jet.data.utils import generate_key
from jet.logger import logger
from jet.models.model_registry.base import BaseModelRegistry

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

WhisperModelsType = Literal[
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "distil-large-v3",
    "distil-large-v3.5",
    "large-v3-turbo",
    "turbo",
]


class WhisperModelRegistry(BaseModelRegistry):
    _models = {}

    @staticmethod
    def load_model(
        model_size: WhisperModelsType = "small",
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: Optional[dict] = None,
        **model_kwargs
    ) -> WhisperModel:
        """
        Static method to load or retrieve a WhisperModel instance.
        """
        instance = WhisperModelRegistry()
        return instance._load_model(
            model_size=model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
            files=files,
            **model_kwargs
        )

    def _load_model(
        self,
        model_size: WhisperModelsType = "large-v3",
        device: str = "cpu",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: Optional[dict] = None,
        **model_kwargs
    ) -> WhisperModel:
        # resolved_model_id = resolve_model_value(model_size)
        resolved_model_id = model_size

        cache_key = generate_key(
            resolved_model_id, device, device_index, compute_type, cpu_threads, num_workers
        )

        if cache_key in self._models:
            logger.info(
                f"Reusing cached Whisper model {resolved_model_id} on {device} "
                f"(compute_type={compute_type}, device_index={device_index}, cpu_threads={cpu_threads}, num_workers={num_workers})"
            )
            return self._models[cache_key]

        logger.info(
            f"Loading Whisper model {resolved_model_id} on {device} "
            f"(compute_type={compute_type}, device_index={device_index}, cpu_threads={cpu_threads}, num_workers={num_workers})"
        )

        try:
            model = WhisperModel(
                model_size_or_path=resolved_model_id,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=download_root,
                local_files_only=local_files_only,
                files=files,
                **model_kwargs,
            )
            self._models[cache_key] = model
            return model
        except Exception as e:
            logger.error(
                f"Failed to load Whisper model {resolved_model_id}: {e}")
            raise ValueError(
                f"Could not load Whisper model {resolved_model_id}: {e}")

    def get_tokenizer(self, model_size: str) -> Optional[object]:
        logger.warning("WhisperModel does not provide a tokenizer instance")
        return None

    def get_config(self, model_size: str) -> Optional[object]:
        logger.warning("WhisperModel does not provide a config instance")
        return None

    def clear(self) -> None:
        self._models.clear()
        logger.info("WhisperModel registry cleared")

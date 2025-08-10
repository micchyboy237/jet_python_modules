from typing import Literal, Optional, Union, List
from faster_whisper import WhisperModel
import torch

from jet.data.utils import generate_key
from jet.logger import logger
from jet.models.utils import resolve_model_value
from jet.models.model_registry.base import BaseModelRegistry, ModelFeatures

WhisperModelsType = Literal[
    "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "distil-small.en", "medium", "medium.en", "distil-medium.en",
            "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-large-v3", "large-v3-turbo", "turbo"
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
        # resolved_model_id = resolve_model_value(model_size)
        resolved_model_id = model_size

        cache_key = generate_key(
            resolved_model_id, compute_type, device_index, cpu_threads, num_workers
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

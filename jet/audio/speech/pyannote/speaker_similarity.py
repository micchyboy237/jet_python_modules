# speaker_similarity.py
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from pyannote.audio import Inference, Model

from jet.logger import logger


class SpeakerSimilarity:
    """
    Reusable class to extract speaker embeddings with pyannote/embedding
    and compute cosine similarity between two audio inputs.

    Inputs: file path (str | Path) or raw waveform (np.ndarray mono float32).
    Uses whole-audio pooling for one fixed-size embedding per input.
    """

    MODEL_REPO = "pyannote/embedding"
    EXPECTED_DIM = 512

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
            hf_token: Hugging Face read token (required)
            device: "cpu" | "cuda" | "mps" | torch.device | None (auto-select if None)
        """
        # Use env variable if not provided.
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")

        # Auto-select the best device if device is None
        if device is None:
            if torch.cuda.is_available():
                selected_device = torch.device("cuda")
                logger.info("Using CUDA (GPU) for inference.")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                selected_device = torch.device("mps")
                logger.info("CUDA not available; using MPS (Apple Silicon) for inference.")
            else:
                selected_device = torch.device("cpu")
                logger.info("CUDA/MPS not available; using CPU for inference.")
        elif isinstance(device, torch.device):
            selected_device = device
        else:
            # string or otherwise
            dev_lower = str(device).lower()
            if dev_lower == "cuda" and torch.cuda.is_available():
                selected_device = torch.device("cuda")
            elif dev_lower == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                selected_device = torch.device("mps")
            elif dev_lower == "cpu":
                selected_device = torch.device("cpu")
            else:
                selected_device = torch.device("cpu")
                logger.info(f"Device '{device}' not available; falling back to CPU.")

        self.device = selected_device
        logger.info(f"Loading {self.MODEL_REPO} on {self.device}...")

        # Load model (requires accepting conditions on HF model page)
        model = Model.from_pretrained(
            self.MODEL_REPO,
            use_auth_token=hf_token
        )
        if model is None:
            raise RuntimeError("Failed to load pyannote/embedding model.")

        # Inference with whole-file pooling
        self.inference = Inference(model, window="whole")

        if self.device.type in ("cuda", "mps"):
            self.inference.to(self.device)
            logger.info(f"Inference moved to {self.device.type.upper()}")
        # No action needed if CPU

    def get_embedding(self, input_data: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Extract normalized speaker embedding.

        Args:
            input_data: Audio file path or mono waveform (np.float32)

        Returns:
            (512,) np.float64 unit vector
        """
        if isinstance(input_data, (str, Path)):
            logger.debug(f"Processing file: {input_data}")
            emb = self.inference(str(input_data))
        elif isinstance(input_data, np.ndarray):
            logger.debug("Processing numpy waveform")
            if input_data.ndim > 1:
                if input_data.shape[0] == 1:
                    input_data = input_data.squeeze(0)
                elif input_data.shape[1] == 1:
                    input_data = input_data.squeeze(1)
                else:
                    raise ValueError("Waveform must be mono (1 channel)")
            emb = self.inference(input_data)
        else:
            raise TypeError("input_data must be str, Path, or np.ndarray (mono waveform)")

        # emb is (1, dim) → squeeze and normalize
        emb = np.squeeze(emb)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        else:
            logger.warning("Zero-norm embedding detected")
        return emb.astype(np.float64)

    def similarity(self, input1: Union[str, Path, np.ndarray], input2: Union[str, Path, np.ndarray]) -> float:
        """
        Cosine similarity between two inputs. Higher = more similar speakers.

        Returns:
            float in [-1, 1]
        """
        emb1 = self.get_embedding(input1)
        emb2 = self.get_embedding(input2)

        sim = float(np.dot(emb1, emb2))
        logger.debug(f"Cosine similarity: {sim:.4f}")
        return sim

    def cosine_distance(self, input1: Union[str, Path, np.ndarray], input2: Union[str, Path, np.ndarray]) -> float:
        """Cosine distance (lower = more similar). Equals 1 - similarity."""
        return 1.0 - self.similarity(input1, input2)

    def assign_speaker_labels(
        self,
        inputs: list[Union[str, Path, np.ndarray]],
        threshold: float = 0.78,
    ) -> tuple[list[int], list[np.ndarray]]:
        """
        Assign integer speaker labels (0, 1, 2, ...) to a list of audio inputs using greedy clustering.
        Reuses self.similarity for every comparison.

        Clustering rule:
        - First input → label 0
        - For each subsequent input, find the existing cluster whose representative
          gives the highest similarity.
        - If that max similarity ≥ threshold → assign to that cluster
        - Else → assign new label

        Args:
            inputs: List of ≥2 audio items (paths or mono np.float32 waveforms)
            threshold: Cosine similarity threshold above which two inputs are considered
                       the same speaker. Tune between ~0.70–0.85 depending on audio quality.

        Returns:
            (labels, embeddings)
            - labels: list of int, same length as inputs
            - embeddings: list of computed normalized embeddings (for inspection)

        Raises:
            ValueError if len(inputs) < 2
        """
        if len(inputs) < 2:
            raise ValueError("assign_speaker_labels requires at least 2 inputs")

        # Cache embeddings
        embeddings: list[np.ndarray] = [self.get_embedding(inp) for inp in inputs]

        labels: list[int] = [0] * len(inputs)
        # Keep index of first member of each cluster as representative
        representatives: list[int] = [0]  # cluster 0 rep = item 0
        next_label = 1

        for i in range(1, len(inputs)):
            max_sim = -1.0
            best_cluster = -1

            for rep_idx in representatives:
                sim = self.similarity(inputs[i], inputs[rep_idx])
                if sim > max_sim:
                    max_sim = sim
                    best_cluster = labels[rep_idx]

            if max_sim >= threshold:
                labels[i] = best_cluster
            else:
                labels[i] = next_label
                representatives.append(i)
                next_label += 1

        return labels, embeddings
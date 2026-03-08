from __future__ import annotations

import torch
import numpy as np


class SileroVAD:
    """Thin streaming wrapper around Silero VAD"""

    def __init__(self, samplerate: int = 16000, device: str | None = None) -> None:
        if samplerate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 Hz or 16000 Hz")

        self.samplerate = samplerate

        self.device = torch.device(
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading Silero VAD on {self.device}... ", end="", flush=True)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.to(self.device)
        self.model.eval()

        torch.set_num_threads(1)
        print("done.")

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if chunk.ndim != 1:
            raise ValueError("Expected 1D audio chunk")

        tensor = torch.from_numpy(chunk).float().to(self.device).unsqueeze(0)
        prob = self.model(tensor, self.samplerate).item()
        return prob

from __future__ import annotations

import torch
import numpy as np


class SpeechBrainVADWrapper:
    """Streaming-like wrapper for speechbrain vad-crdnn-libriparty"""

    def __init__(self, device: str | None = None) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        print(f"Loading SpeechBrain VAD on {self.device}... ", end="", flush=True)

        from speechbrain.inference.VAD import VAD

        self.vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
            run_opts={"device": str(self.device)},
        )
        self.vad.eval()
        print("done.")

        self.sample_rate = 16000
        self.context_samples = int(0.5 * self.sample_rate)
        self.audio_ring = torch.zeros(
            self.context_samples, dtype=torch.float32, device=self.device
        )
        self.write_pos = 0

    @torch.inference_mode()
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return 0.0

        chunk_t = torch.from_numpy(chunk).float().to(self.device)
        chunk_len = len(chunk_t)

        space = self.context_samples - self.write_pos
        if chunk_len <= space:
            self.audio_ring[self.write_pos : self.write_pos + chunk_len] = chunk_t
            self.write_pos += chunk_len
        else:
            self.audio_ring[:chunk_len] = chunk_t[-self.context_samples :]
            self.write_pos = chunk_len

        prob_tensor = self.vad.get_speech_prob_chunk(self.audio_ring.unsqueeze(0))
        return float(prob_tensor[-1, -1].item())

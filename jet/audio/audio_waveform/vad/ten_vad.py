from typing import override

import numpy as np
from jet.audio.audio_waveform.vad.base_vad import BaseVAD
from ten_vad import TenVad


class TenVadWrapper(BaseVAD):
    """
    Simple wrapper so TenVad works the same way as other VAD models
    in the visualizer (has get_speech_prob method).
    """

    def __init__(self, hop_size: int = 160, threshold: float = 0.5):
        self.vad = TenVad(hop_size=hop_size, threshold=threshold)
        self.last_prob = 0.0

    @override
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Takes a block of audio samples and returns speech probability [0.0 - 1.0].
        Compatible with VADObserver.
        """
        if len(chunk) == 0:
            return self.last_prob

        hop_size = self.vad.hop_size
        if len(chunk) > hop_size:
            chunk = chunk[:hop_size]
        else:
            pass  # chunk already refers to the full input

        if chunk.dtype != np.int16:
            chunk_int16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)
        else:
            chunk_int16 = chunk

        try:
            prob, _ = self.vad.process(chunk_int16)
            self.last_prob = float(prob)
        except Exception:
            # Fallback on error
            pass

        return self.last_prob

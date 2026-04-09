import numpy as np
from ten_vad import TenVad


class TenVadWrapper:
    """
    Simple wrapper so TenVad works the same way as other VAD models
    in the visualizer (has get_speech_prob method).
    """

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        self.vad = TenVad(hop_size=hop_size, threshold=threshold)
        self.last_prob = 0.0

    def get_speech_prob(self, samples: np.ndarray) -> float:
        """
        Takes a block of audio samples and returns speech probability [0.0 - 1.0].
        Compatible with VADObserver.
        """
        if len(samples) == 0:
            return self.last_prob

        # TenVad expects exactly hop_size samples (default 256) of int16
        # If block_size from AudioStreamManager is different (512), we take first hop_size
        hop_size = self.vad.hop_size
        if len(samples) > hop_size:
            chunk = samples[:hop_size]
        else:
            chunk = samples

        # Convert float32 → int16 (required by TenVad)
        if samples.dtype != np.int16:
            # Simple scaling: assume samples are in [-1.0, 1.0] range
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

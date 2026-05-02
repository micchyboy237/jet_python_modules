# jet_python_modules/jet/audio/audio_waveform/vad/base_vad.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVAD(ABC):
    """
    Abstract base interface for all VAD (Voice Activity Detection) wrappers.

    Every concrete VAD class must implement get_speech_prob, which accepts
    a 1-D float32 numpy array of audio samples (mono, 16 kHz) and returns
    a speech probability in [0.0, 1.0].

    Type contract (mirrored from FireRedVADWrapper):
        chunk : np.ndarray  — 1-D array, dtype float32, 16 kHz mono samples
        return: float       — speech probability in [0.0, 1.0]
    """

    @abstractmethod
    def get_speech_prob(self, chunk: np.ndarray) -> float:
        """
        Process one chunk of audio and return a speech probability.

        Parameters
        ----------
        chunk:
            1-D float32 numpy array of mono 16 kHz audio samples.
            May be empty (length 0); implementations must handle that
            gracefully, typically by returning the last known probability.

        Returns
        -------
        float
            Speech probability in the range [0.0, 1.0].
        """

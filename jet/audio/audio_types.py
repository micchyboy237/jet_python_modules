# audio_types.py

from __future__ import annotations
from pathlib import Path
from typing import Union

import torch
import numpy as np

AudioInput = Union[str, Path, np.ndarray, torch.Tensor, bytes]

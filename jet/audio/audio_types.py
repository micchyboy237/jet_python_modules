# audio_types.py

from __future__ import annotations
from typing import Union

import os
import torch
import numpy as np
import numpy.typing as npt

# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]

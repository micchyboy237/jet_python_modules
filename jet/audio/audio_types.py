# audio_types.py

from __future__ import annotations

import os
from typing import Union

import numpy as np
import numpy.typing as npt
import torch

AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

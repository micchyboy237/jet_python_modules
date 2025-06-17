import shutil
from typing import Optional, Union, Callable
import logging
import os
from pathlib import Path
from mlx_lm import convert
import mlx.nn as nn

logger = logging.getLogger(__name__)


def convert_hf_to_mlx(
    hf_path: str,
    weights_dir: str,
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    quant_predicate: Optional[Union[Callable[[
        str, nn.Module, dict], Union[bool, dict]], str]] = None,
    revision: Optional[str] = None,
    overwrite: bool = False,
    model_id: Optional[str] = None,  # New parameter to pass the model_id
) -> None:
    try:
        logger.info(
            f"Converting Hugging Face model from {hf_path} to MLX format at {weights_dir}")
        safetensors_index = os.path.join(
            hf_path, "model.safetensors.index.json")
        safetensors_file = os.path.join(hf_path, "model.safetensors")
        if not (os.path.exists(safetensors_file) or os.path.exists(safetensors_index)):
            logger.error(f"No safetensors files found in {hf_path}")
            raise FileNotFoundError(
                f"No model.safetensors or model.safetensors.index.json found in {hf_path}"
            )
        if overwrite and os.path.exists(weights_dir):
            shutil.rmtree(weights_dir)
        # Use model_id as hf_path for the convert function if provided, else fallback to hf_path
        convert(
            hf_path=model_id or hf_path,
            mlx_path=weights_dir,
            quantize=quantize,
            q_group_size=q_group_size,
            q_bits=q_bits,
            dtype=dtype,
            upload_repo=None,
            revision=revision,
            dequantize=False,
            quant_predicate=quant_predicate,
        )
        weights_path = os.path.join(weights_dir, "weights.npz")
        if not os.path.exists(weights_path):
            logger.error(f"Failed to generate weights.npz at {weights_path}")
            raise FileNotFoundError(f"weights.npz not found in {weights_dir}")
        logger.info(f"Successfully generated weights.npz at {weights_path}")
    except ValueError as e:
        logger.error(f"Conversion failed: {str(e)}")
        raise ValueError(f"Could not convert model to MLX format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {str(e)}")
        raise ValueError(
            f"Unexpected error converting model to MLX format: {str(e)}")

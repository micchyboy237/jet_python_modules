import json
import numpy as np
from jet.logger import logger


def check_numpy_config() -> dict:
    """Check BLAS configuration for NumPy."""
    config = np.__config__.show(mode='dicts')
    logger.gray("\nNumpy Config:")
    logger.info(json.dumps(config, indent=2))
    return config


def check_accelerate_usage() -> dict:
    """Check BLAS configuration for NumPy."""
    config = np.__config__.show(mode='dicts')
    blas_info = config.get('Build Dependencies', {}).get('blas', {})
    logger.gray("\nBLAS Info:")
    logger.info(json.dumps(blas_info, indent=2))
    return blas_info

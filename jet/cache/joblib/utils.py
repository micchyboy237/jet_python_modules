import os
from jet.transformers.object import make_serializable
import joblib
from jet.logger import logger
from typing import Any, Type, TypeVar, Optional, TypedDict

from pydantic.main import BaseModel


def save_cache(file_path: str, data: Any) -> None:
    """
    Save data to a file using joblib.

    :param file_path: Path to save the file.
    :param data: Python object to be saved.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Serialize the model to a dictionary
    joblib.dump(make_serializable(data), file_path)
    logger.success(f"Data saved successfully to {file_path}")


def load_cache(file_path: str) -> Any:
    """
    Load data from a file using joblib.

    :param file_path: Path to the saved file.
    :return: The loaded Python object, or None if an error occurs.
    """
    data: Any = joblib.load(file_path)
    logger.orange(f"Data loaded successfully from {file_path}")
    return data


def load_or_save_cache(
    file_path: str,
    data: Optional[Any | BaseModel] = None,
    model: Optional[Type[BaseModel]] = None
) -> BaseModel:
    """
    Load data from or save data to a cache file, with the model type determined at runtime.
    Creates directories if they do not exist.

    Args:
        file_path (str): The path to the cache file (must end with '.pkl').
        data (Optional[BaseModel]): The data to save. If None, the function loads data instead.
        model (Optional[Type[BaseModel]]): The Pydantic model class used to validate the data when loading. 
                                           If not provided, the function will not attempt to load data.

    Returns:
        BaseModel: The loaded data if data is None and model is provided; otherwise, the saved data.

    Raises:
        ValueError: If the file path does not end with '.pkl', or if loading data without providing a model.
    """
    if os.path.exists(file_path):
        return load_cache(file_path)
    elif data:
        save_cache(file_path, data)
        return data


def load_from_cache_or_compute(func, *args, file_path: str = "", use_cache: bool = True, **kwargs):
    """
    Caches the result of a function to a .pkl file or computes it if cache doesn't exist.

    Parameters:
    - func: Function to compute the result.
    - args: Positional arguments for the function.
    - use_cache: Bool, whether to use the cache.
    - file_path: Path to the cache file.
    - kwargs: Keyword arguments for the function.

    Returns:
    - Cached or computed result.
    """
    if not file_path.endswith(".pkl"):
        raise ValueError("Cache file must have a .pkl extension.")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if use_cache and os.path.exists(file_path):
        logger.success(f"Cache hit! File: {file_path}")
        return joblib.load(file_path)

    # Compute the result and cache it
    logger.info(f"Cache miss! Computing result for: {file_path}")
    result = func(*args, **kwargs)
    joblib.dump(result, file_path)
    logger.success(f"Saved cache to: {file_path}")
    return result


__all__ = [
    "save_cache",
    "load_cache",
    "load_or_save_cache",
    "load_from_cache_or_compute",
]


# Example Usage
if __name__ == "__main__":
    class MyCacheModel(BaseModel):
        key: str
        value: int

    cache_file = "generated/example.pkl"
    data = MyCacheModel(key="example", value=42)

    # Save data
    saved_data = load_or_save_cache(cache_file, data=data)
    logger.debug(saved_data)  # Output: key='example' value=42

    # Load data
    loaded_data = load_or_save_cache(cache_file, model=MyCacheModel)
    logger.success(loaded_data)  # Output: key='example' value=42

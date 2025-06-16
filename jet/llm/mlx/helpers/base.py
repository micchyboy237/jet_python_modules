from typing import List, Union, Callable, Optional
from transformers import PreTrainedTokenizer, BatchEncoding
import numpy as np
from jet.llm.mlx.mlx_types import ModelType
from typing import Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from jet.llm.mlx.mlx_types import EmbedModelType, MLXTokenizer, LLMModelType, ModelType
from jet.llm.mlx.models import AVAILABLE_MODELS, resolve_model, resolve_model_key
from jet.llm.mlx.models import resolve_model_value

import mlx.nn as nn
import torch

device = torch.device("cpu")


def load_model(model: ModelType):
    model_key = resolve_model_key(model)
    if model in AVAILABLE_MODELS:
        return load_llm_model(model)
    else:
        return load_embed_model(model)


def load_llm_model(model: LLMModelType) -> Tuple[nn.Module, MLXTokenizer]:
    from mlx_lm import load

    model_path = str(resolve_model(model))

    return load(model_path)


def load_embed_model(model_name: EmbedModelType):
    model_id = resolve_model_value(model_name)
    model = SentenceTransformer(model_id)
    model = model.to(device)  # Move model to MPS or CPU
    return model


def get_system_date_prompt():
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"


__all__ = [
    "load_model",
    "load_llm_model",
    "load_embed_model",
    "get_system_date_prompt",
]

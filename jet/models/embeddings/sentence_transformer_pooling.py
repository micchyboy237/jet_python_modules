# File: sentence_transformer_pooling.py
import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sentence_transformer(
    model_name: str,
    pooling_mode: str = "mean_tokens",
    use_mps: bool = True
) -> SentenceTransformer:
    """Load a SentenceTransformer with specified pooling mode."""
    try:
        logger.debug(
            f"Loading model {model_name} with pooling mode {pooling_mode}")
        device = "cpu"
        logger.debug(f"Using device: {device}")

        # Load transformer model and tokenizer
        transformer_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure pooling
        pooling_config = {
            "word_embedding_dimension": transformer_model.config.hidden_size,
            "pooling_mode_cls_token": pooling_mode == "cls_token",
            "pooling_mode_mean_tokens": pooling_mode == "mean_tokens",
            "pooling_mode_max_tokens": pooling_mode == "max_tokens",
            "pooling_mode_mean_sqrt_len_tokens": pooling_mode == "mean_sqrt_len_tokens"
        }
        pooling_layer = Pooling(
            word_embedding_dimension=pooling_config["word_embedding_dimension"],
            pooling_mode_cls_token=pooling_config["pooling_mode_cls_token"],
            pooling_mode_mean_tokens=pooling_config["pooling_mode_mean_tokens"],
            pooling_mode_max_tokens=pooling_config["pooling_mode_max_tokens"],
            pooling_mode_mean_sqrt_len_tokens=pooling_config["pooling_mode_mean_sqrt_len_tokens"]
        )

        # Create SentenceTransformer
        model = SentenceTransformer(
            modules=[transformer_model, pooling_layer], device=device)
        logger.info(
            f"Successfully loaded model {model_name} with {pooling_mode} pooling on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise


def encode_sentences(model: SentenceTransformer, sentences: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode sentences into embeddings."""
    try:
        logger.debug(
            f"Encoding {len(sentences)} sentences with batch size {batch_size}")
        embeddings = model.encode(
            sentences, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        logger.info(
            f"Encoded {len(sentences)} sentences into {embeddings.shape} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Encoding failed: {str(e)}")
        raise

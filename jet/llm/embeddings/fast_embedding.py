from jet.logger import logger
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import math


def generate_embeddings(
    documents: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 128,
    max_length: int = 512,
    use_mps: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of documents using a SentenceTransformer model.

    Args:
        documents: List of documents to encode
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding
        max_length: Maximum token length for the model
        use_mps: Whether to use MPS (Apple Silicon GPU) if available
        normalize: Whether to normalize embeddings

    Returns:
        numpy array of shape (n_documents, embedding_dim)
    """
    try:
        # Initialize device
        device = 'mps' if use_mps and torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Load model
        model = SentenceTransformer(model_name)
        model = model.to(device)
        model.eval()  # Set to evaluation mode

        # Enable mixed precision if using MPS
        if device == 'mps':
            torch.set_default_dtype(torch.float16)

        # Calculate number of batches
        n_batches = math.ceil(len(documents) / batch_size)
        embeddings = []

        # Process batches with progress bar
        for i in tqdm(range(0, len(documents), batch_size), total=n_batches, desc="Encoding documents"):
            batch = documents[i:i + batch_size]

            # Encode batch
            with torch.no_grad():
                batch_embeddings = model.encode(
                    batch,
                    batch_size=batch_size,
                    max_length=max_length,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        # Reset default dtype
        if device == 'mps':
            torch.set_default_dtype(torch.float32)

        return embeddings

    except Exception as e:
        logger.error(f"Error during embedding generation: {str(e)}")
        raise

    finally:
        # Clear memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.mps.empty_cache() if device == 'mps' else None

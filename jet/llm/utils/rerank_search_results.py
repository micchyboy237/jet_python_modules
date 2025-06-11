import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc
from jet.logger import logger
from jet.llm.utils.transformer_embeddings import SimilarityResult


def rerank_search_results(
    query: str,
    results: List[SimilarityResult],
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    batch_size: Optional[int] = 32,
    top_k: Optional[int] = None,
    ids: Optional[List[str]] = None
) -> List[SimilarityResult]:
    """
    Rerank a list of SimilarityResult objects using a cross-encoder model.

    Args:
        query: The query string used for reranking.
        results: List of SimilarityResult objects from initial search.
        rerank_model: Cross-encoder model for reranking.
        batch_size: Batch size for reranking (default: 32, suitable for Mac M1).
        top_k: Number of top results to return after reranking (default: None, returns all).
        ids: Optional list of document IDs to override existing IDs (default: None).

    Returns:
        List of SimilarityResult objects with updated scores, ranks, and IDs.
    """
    if not query or not results:
        logger.warning(
            "Empty query or results list provided. Returning empty list.")
        return []

    # Validate ids if provided
    if ids is not None:
        if len(ids) != len(results):
            logger.error(
                f"Length of ids ({len(ids)}) must match length of results ({len(results)})")
            raise ValueError(
                f"Length of ids ({len(ids)}) must match length of results ({len(results)})")
        if len(ids) != len(set(ids)):
            logger.error("IDs must be unique")
            raise ValueError("IDs must be unique")

    # Set top_k to len(results) if not specified
    if top_k is None:
        top_k = len(results)
    top_k = min(top_k, len(results))

    # Validate top_k
    if top_k <= 0:
        logger.warning("top_k must be positive. Returning empty list.")
        return []

    # Determine device (MPS for Mac M1, CUDA if available, else CPU)
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")

    # Load cross-encoder model and tokenizer
    try:
        rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        rerank_model_instance = AutoModelForSequenceClassification.from_pretrained(
            rerank_model, torch_dtype=torch.float16
        ).to(device)
        rerank_model_instance.eval()
    except Exception as e:
        logger.error(f"Failed to load rerank model {rerank_model}: {str(e)}")
        raise RuntimeError(f"Failed to load rerank model: {str(e)}")

    # Prepare query-document pairs
    pairs = [(query, result["text"]) for result in results]

    # Define dataset and dataloader
    class PairDataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs[idx]

    def collate_fn(batch):
        return batch  # Return batch as list of tuples

    dataset = PairDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # Compute reranking scores
    rerank_scores = []
    with torch.autocast(device_type=device, dtype=torch.float16):
        for batch in tqdm(dataloader, desc="Reranking", leave=True):
            queries, docs = zip(*batch)
            inputs = rerank_tokenizer(
                list(queries),
                list(docs),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = rerank_model_instance(**inputs)
                scores = outputs.logits.squeeze(-1)
                scores = torch.sigmoid(scores)  # Convert to [0,1] range
                rerank_scores.extend(scores.cpu().tolist())

            # Clean up
            del inputs, outputs, scores
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Update results with reranking scores and IDs
    for i, score in enumerate(rerank_scores):
        results[i]["score"] = float(score)
        if ids is not None:
            results[i]["id"] = ids[i]

    # Sort by reranking score and take top_k
    sorted_results = sorted(
        results, key=lambda x: x["score"], reverse=True)[:top_k]

    # Update ranks
    for rank, result in enumerate(sorted_results, start=1):
        result["rank"] = rank

    return sorted_results

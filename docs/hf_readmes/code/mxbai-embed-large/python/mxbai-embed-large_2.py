from typing import Dict, List, Optional, TypedDict
import uuid

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest score).
        doc_index: Index of the document in the input list.
        score: Similarity score between query and text.
        percent_difference: Percentage difference from the highest score, rounded to 2 decimals.
        text: The compared text.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    percent_difference: Optional[float]
    text: str

# For retrieval you need to pass this prompt. Please find our more in our blog post.


def transform_query(query: str) -> str:
    """ For retrieval, add the prompt for query (not for documents).
    """
    return f'Represent this sentence for searching relevant passages: {query}'

# The model works really well with cls pooling (default) but also with mean pooling.


def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


def compute_similarity_results(embeddings: np.ndarray, docs: List[str], query_index: int = 0) -> List[SimilarityResult]:
    """Compute similarity results for given embeddings and documents."""
    similarities = cos_sim(
        embeddings[query_index], embeddings[1:]).flatten().tolist()
    similarity_results: List[SimilarityResult] = []
    # Avoid division by zero
    max_score = max(similarities) if similarities else 1.0

    for idx, (score, text) in enumerate(zip(similarities, docs[1:])):
        percent_diff = round(((max_score - score) / max_score)
                             * 100, 2) if max_score != 0 else 0.0
        similarity_results.append({
            "id": str(uuid.uuid4()),
            "rank": 0,  # Will be updated after sorting
            "doc_index": idx + 1,  # Index in docs list (skip query at index 0)
            "score": round(score, 4),
            "percent_difference": percent_diff,
            "text": text
        })

    # Sort by score in descending order and assign ranks
    similarity_results.sort(key=lambda x: x["score"], reverse=True)
    for rank, result in enumerate(similarity_results, 1):
        result["rank"] = rank

    return similarity_results


# 1. Load model
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# Check if MPS is available and move model to MPS device, else use CPU
device = torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)

docs = [
    transform_query('A man is eating a piece of bread'),
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# 2. Encode inputs once
inputs = tokenizer(docs, padding=True, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs).last_hidden_state

# 3. Compute embeddings and similarities for both pooling strategies
strategies = ['cls', 'mean']
for strategy in strategies:
    embeddings = pooling(outputs, inputs, strategy)
    similarity_results = compute_similarity_results(embeddings, docs)

    # Print results for the current strategy
    print(f"\nSimilarity Results (Pooling Strategy: {strategy.upper()}):")
    for result in similarity_results:
        print(f"ID: {result['id']}")
        print(f"Rank: {result['rank']}")
        print(f"Doc Index: {result['doc_index']}")
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Percent Difference: {result['percent_difference']}%")
        print()


# Compare CLS and Mean rankings
cls_results = compute_similarity_results(pooling(outputs, inputs, 'cls'), docs)
mean_results = compute_similarity_results(
    pooling(outputs, inputs, 'mean'), docs)

print("\nComparison of CLS vs Mean Pooling Rankings:")
for cls_res, mean_res in zip(cls_results, mean_results):
    print(f"Text: {cls_res['text']}")
    print(f"CLS Rank: {cls_res['rank']}, Score: {cls_res['score']}")
    print(f"Mean Rank: {mean_res['rank']}, Score: {mean_res['score']}")
    print()

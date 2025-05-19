from typing import Literal, get_args
import json
import os
import numpy as np
from datasets import Dataset
from jet.libs.txtai.vectors import VectorsFactory
from jet.libs.txtai.ann import ANNFactory
from jet.libs.txtai.scoring import ScoringFactory
from jet.logger import logger


ScoringMethod = Literal["bm25", "sif", "pgtext", "tfidf"]


def calculate_tokens(text: str) -> int:
    """Calculate the token length of the text."""
    # return len(tokenizer.encode(text))
    return len(text)


def load_local_json(filepath):
    """Load a JSON file into a Hugging Face dataset."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return Dataset.from_dict({"id": [item["id"] for item in data],
                              "tags": [item["metadata"]["tags"] for item in data],
                              "page_content": [item["page_content"] for item in data]})


def load_or_create_embeddings(texts, model, cache_file=None) -> np.ndarray:
    """Load cached embeddings or create new ones if not available."""
    if cache_file and os.path.exists(cache_file):
        logger.debug("Loading embeddings from cache:")
        logger.info(cache_file)
        embeddings = np.load(cache_file)
    else:
        logger.debug("Creating embeddings...")
        embeddings = np.zeros(dtype=np.float32, shape=(len(texts), 384))
        batch, index, batchsize = [], 0, 128
        for text in texts:
            batch.append(text)
            if len(batch) == batchsize:
                vectors = model.encode(batch)
                embeddings[index: index + vectors.shape[0]] = vectors
                index += vectors.shape[0]
                batch = []
        if batch:
            vectors = model.encode(batch)
            embeddings[index: index + vectors.shape[0]] = vectors
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

        # Cache the embeddings if cache_file is provided
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, embeddings)
            print(f"Embeddings saved to cache: {cache_file}")
            logger.log("Embeddings saved to cache:", cache_file,
                       colors=["LOG", "BRIGHT_SUCCESS"])
    return embeddings


def build_ann_index(embeddings):
    """Build an ANN index for the embeddings."""
    ann = ANNFactory.create({"backend": "faiss"})
    ann.index(embeddings)
    return ann


def ann_search(query, ann, model, dataset: Dataset, top_k: int = 3):
    """Search for a query in the ANN index."""
    texts = [row["page_content"] for row in dataset]
    query_vec = model.encode([query])
    query_vec /= np.linalg.norm(query_vec)
    results = ann.search(query_vec, top_k)[0]
    return [
        {
            "id": dataset["id"][uid],
            "tags": dataset["tags"][uid],
            "text": texts[uid],
            "score": score,
            "tokens": calculate_tokens(texts[uid])
        } for uid, score in results
    ]


def scoring_search(query, dataset: Dataset, method: ScoringMethod = "bm25", top_k=3):
    """Perform a BM25-based scoring search."""
    texts = [row["page_content"] for row in dataset]
    scoring = ScoringFactory.create(
        {"method": method, "terms": True, "content": True})
    scoring.index((x, text, None) for x, text in enumerate(texts))
    results = scoring.search(query, top_k)
    return [
        {
            "id": dataset["id"][row["id"]],
            "tags": dataset["tags"][row["id"]],
            "text": row["text"],
            "score": row["score"],
            "tokens": calculate_tokens(row["text"])
        } for row in results
    ]

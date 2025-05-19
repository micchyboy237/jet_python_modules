"""
Embeddings Load Function Module
"""

import os
import torch
from jet.libs.txtai.embeddings import (
    Embeddings as TxtaiEmbeddings,
    Configuration,
    Reducer,
)
from jet.logger import logger


class Embeddings(TxtaiEmbeddings):
    """
    Embeddings engine for semantic search and vector operations.
    Contains the method to load an existing index.
    """

    def load(self, path=None, cloud=None, config=None, **kwargs):
        """
        Loads an existing index from path.

        Args:
            path: input path
            cloud: cloud storage configuration
            config: configuration overrides
            kwargs: additional configuration as keyword args
        """
        # Load from cloud if configured
        cloud = self.createcloud(cloud=cloud, **kwargs)
        if cloud:
            path = cloud.load(path)

        # Check if this is an archive file and extract
        path, apath = self.checkarchive(path)
        if apath:
            self.archive.load(apath)

        # Load index configuration
        self.config = self._load_config(path)

        # Apply config overrides
        self.config = {**self.config, **config} if config else self.config

        # Approximate nearest neighbor index
        self.ann = self.createann()
        if self.ann:
            self.ann.load(f"{path}/embeddings")

        # Dimensionality reduction model (e.g., PCA)
        if self.config.get("pca"):
            self.reducer = self._create_reducer(path)

        # Index IDs (used when content is disabled)
        self.ids = self.createids()
        if self.ids:
            self.ids.load(f"{path}/ids")

        # Document database
        self.database = self.createdatabase()
        if self.database:
            self.database.load(f"{path}/documents")

        # Scoring (e.g., sparse vectors)
        self.scoring = self.createscoring()
        if self.scoring:
            self.scoring.load(f"{path}/scoring")

        # Subindexes
        self.indexes = self.createindexes()
        if self.indexes:
            self.indexes.load(f"{path}/indexes")

        # Graph network
        self.graph = self.creategraph()
        if self.graph:
            self.graph.load(f"{path}/graph")

        # Dense vectors (embeddings)
        self.model = self.loadvectors()

        # Query model
        self.query = self.loadquery()

    def _load_config(self, path: str):
        # Load index configuration
        config = Configuration().load(path)
        return config

    def _create_reducer(self, path: str):
        reducer = Reducer()
        reducer.load(f"{path}/lsa")
        return reducer


def load_or_create_embeddings(dataset, embedding_model, cache_file):
    """
    Load embeddings from cache if available, otherwise create and cache them.
    Args:
    - dataset: The dataset to use for creating embeddings.
    - embedding_model: The model to use for embedding creation.
    - cache_file: The file path to use for caching embeddings.
    Returns:
    - The embeddings object.
    """
    def transform(inputs) -> torch.Tensor:
        from sentence_transformers import SentenceTransformer
        # Initialize the model
        sentence_transformer_ef = SentenceTransformer(embedding_model)
        # Get embeddings for the provided sentences
        embeddings = sentence_transformer_ef.encode(
            inputs, convert_to_tensor=True)
        return embeddings

    embeddings = Embeddings(
        config={"method": "external", "transform": transform, "content": True})
    if os.path.exists(cache_file):
        embeddings.load(path=cache_file)  # Call load on the instance
        logger.info("Loaded cached embeddings.")
    else:
        embeddings.index(stream(dataset, "page_content", 10000))
        embeddings.save(cache_file)
        logger.info("Created and cached embeddings.")
    return embeddings


def stream(dataset, field, limit):
    """
    Streams data from a dataset, yielding tuples of index, text, and None until the limit is reached.
    """
    index = 0
    for row in dataset:
        yield (index, row[field], None)
        index += 1
        if index >= limit:
            break

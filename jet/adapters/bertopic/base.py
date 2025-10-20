from typing import List, Optional, Tuple, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# BERTopic
from bertopic import BERTopic as BaseBERTopic
from bertopic.representation import BaseRepresentation
from jet.adapters.bertopic.embeddings import BERTopicLlamacppEmbedder
from jet.adapters.bertopic.utils import get_vectorizer

DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

class BERTopic(BaseBERTopic):
    def __init__(
        self,
        language: str = "english",
        top_n_words: int = 10,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_topic_size: int = 2,
        nr_topics: Optional[Union[int, str]] = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        seed_topic_list: Optional[List[List[str]]] = None,
        zeroshot_topic_list: Optional[List[str]] = None,
        zeroshot_min_similarity: float = 0.7,
        embedding_model=None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model: Optional[CountVectorizer] = None,
        ctfidf_model: Optional[TfidfTransformer] = None,
        representation_model: Optional[BaseRepresentation] = None,
        verbose: bool = True,
        use_cache: bool = False,
    ):
        if not embedding_model or isinstance(embedding_model, str):
            embedder = BERTopicLlamacppEmbedder(embedding_model or DEFAULT_EMBEDDING_MODEL, use_cache=use_cache)
            embedding_model = embedder
        
        if not vectorizer_model:
            vectorizer_model = get_vectorizer()

        super().__init__(
            language=language,
            top_n_words=top_n_words,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            low_memory=low_memory,
            calculate_probabilities=calculate_probabilities,
            seed_topic_list=seed_topic_list,
            zeroshot_topic_list=zeroshot_topic_list,
            zeroshot_min_similarity=zeroshot_min_similarity,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            verbose=verbose,
        )

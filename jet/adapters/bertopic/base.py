from jet.adapters.bertopic.embeddings import BERTopicLlamacppEmbedder
from jet.adapters.bertopic.utils import get_vectorizer
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# BERTopic
from bertopic import BERTopic as BaseBERTopic
from bertopic.representation import BaseRepresentation

DEFAULT_EMBEDDING_MODEL: LLAMACPP_EMBED_KEYS = "embeddinggemma"


class BERTopic(BaseBERTopic):
    def __init__(
        self,
        language: str = "english",
        top_n_words: int = 10,
        n_gram_range: tuple[int, int] = (1, 3),
        min_topic_size: int = 2,
        nr_topics: int | str | None = None,
        low_memory: bool = False,
        calculate_probabilities: bool = False,
        seed_topic_list: list[list[str]] | None = None,
        zeroshot_topic_list: list[str] | None = None,
        zeroshot_min_similarity: float = 0.7,
        embedding_model: LLAMACPP_EMBED_KEYS | None = None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model: CountVectorizer | None = None,
        ctfidf_model: TfidfTransformer | None = None,
        representation_model: BaseRepresentation | None = None,
        verbose: bool = True,
        use_cache: bool = False,
    ):
        if not embedding_model or isinstance(embedding_model, str):
            embedder = BERTopicLlamacppEmbedder(
                embedding_model or DEFAULT_EMBEDDING_MODEL, use_cache=use_cache
            )
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

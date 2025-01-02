import operator
from typing import Optional, Sequence, Union
from langchain_core.callbacks import Callbacks, CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document, BaseDocumentCompressor

from jet.db.chroma import (
    ChromaClient,
    VectorItem,
    InitialDataEntry,
    SearchResult,
)
from jet.llm.ollama import OllamaEmbeddingFunction
from jet.llm.model import get_model_path
from jet.transformers import make_serializable
from jet.logger import logger

# Defaults
DEFAULT_USE_OLLAMA = True
DEFAULT_USE_RERANKER = True
DEFAULT_COLLECTION_NAME = "jet_default_collection"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_RERANK_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 10
DEFAULT_RERANK_THRESHOLD = 0.3
DEFAULT_OVERWRITE = False


class RerankerRetriever():
    def __init__(
        self,
        data: list[str] | list[InitialDataEntry],
        *,
        use_ollama: bool = DEFAULT_USE_OLLAMA,
        use_reranker: bool = DEFAULT_USE_RERANKER,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        overwrite: str = DEFAULT_OVERWRITE,
    ):
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.embed_batch_size = embed_batch_size
        self.use_reranker = use_reranker
        self.rerank_model = rerank_model

        # Setup embedding function
        if use_ollama:
            self.embedding_function = OllamaEmbeddingFunction(
                model_name=self.embed_model,
                batch_size=self.embed_batch_size,
            )
        else:
            self.embedding_function = get_model_path(self.embed_model)

        # Setup Vector DB
        self.db = ChromaClient(
            self.collection_name,
            self.embedding_function,
            initial_data=data,
            overwrite=overwrite,
        )

        if not use_ollama:
            from sentence_transformers import CrossEncoder

            self.reranking_function = CrossEncoder(
                get_model_path(self.rerank_model, True),
                trust_remote_code=True,
            )
        else:
            self.reranking_function = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
        limit: Optional[int] = None,
    ) -> list[Document]:
        limit = limit or DEFAULT_TOP_K
        result = self.db.query(
            collection_name=self.collection_name,
            vectors=[self.embedding_function(query)],
            limit=limit,
        )

        ids = result["ids"]
        metadatas = result["metadatas"]
        documents = result["documents"]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results

    def search(
        self, query: Union[str, list[str]], embeddings: Optional[list[float]] = None
    ) -> list[SearchResult]:
        return self.db.search(texts=query, embeddings=embeddings)

    def search_with_reranking(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_threshold: Optional[float] = None,
    ) -> dict:
        top_k = top_k or DEFAULT_TOP_K
        rerank_threshold = rerank_threshold or DEFAULT_RERANK_THRESHOLD

        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
        from jet.db.chroma import convert_search_results

        try:
            result = self.db.get()

            if "metadatas" in result and result["metadatas"]:
                result["metadatas"] = [metadata or {}
                                       for metadata in result["metadatas"]]

            bm25_retriever = BM25Retriever.from_texts(
                texts=result['documents'],
                metadatas=result['metadatas'],
            )
            bm25_retriever.k = top_k

            vector_search_retriever = VectorSearchRetriever(
                db=self.db,
                embedding_function=self.embedding_function,
                top_k=top_k,
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever,
                            vector_search_retriever], weights=[0.5, 0.5]
            )
            compressor = RerankCompressor(
                embedding_function=self.embedding_function,
                top_n=top_k,
                reranking_function=self.reranking_function,
                r_score=rerank_threshold,
            )

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )

            result = compression_retriever.invoke(query)
            result = {
                "ids": [[d.id for d in result]],
                "distances": [[d.metadata.get("score") for d in result]],
                "documents": [[d.page_content for d in result]],
                "metadatas": [[d.metadata for d in result]],
            }

            search_results = convert_search_results(result)

            logger.info(
                "query_doc_with_hybrid_search:result "
                + f'{result["metadatas"]} {result["distances"]}'
            )
            return search_results
        except Exception as e:
            logger.error(f"Error in search_with_reranking: {str(e)}")
            raise


class VectorSearchRetriever(BaseRetriever):
    db: ChromaClient
    embedding_function: any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        result = self.db.search(
            embeddings=[self.embedding_function(query)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


class RerankCompressor(BaseDocumentCompressor):
    def __init__(self, embedding_function, top_n, reranking_function, r_score):
        self.embedding_function = embedding_function
        self.top_n = top_n
        self.reranking_function = reranking_function
        self.r_score = r_score

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(zip(documents, scores.tolist()))
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores,
                        key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results


__all__ = [
    "RerankerRetriever",
    "VectorSearchRetriever",
    "RerankCompressor",
]

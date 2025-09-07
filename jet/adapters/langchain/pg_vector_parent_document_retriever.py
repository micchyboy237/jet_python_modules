from jet.db.postgres.pgvector import PgVectorClient
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import List, Optional, Dict, Any, Tuple
from jet.logger import CustomLogger


class PgVectorParentDocumentRetriever(VectorStore):
    """Custom retriever for parent-child document retrieval using pgvector."""

    def __init__(
        self,
        client: PgVectorClient,
        embedding_model,
        child_splitter,
        database_name: str,
        collection_name: str,
        parent_table_name: str,
        child_table_name: str,
        text_key: str = "page_content",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.child_splitter = child_splitter
        self.database_name = database_name
        self.collection_name = collection_name
        self.parent_table_name = parent_table_name
        self.child_table_name = child_table_name
        self.text_key = text_key
        self.search_kwargs = search_kwargs or {"top_k": 10}
        self.dimension = 1536  # Default dimension for mxbai-embed-large
        self.client._ensure_table_exists(
            self.parent_table_name, self.dimension)
        self.client._ensure_table_exists(self.child_table_name, self.dimension)

    async def aadd_documents(self, documents: List[Document]) -> None:
        """Asynchronously add documents to parent and child tables."""
        parent_rows = []
        child_rows = []
        for doc in documents:
            parent_id = self.client.generate_unique_hash()
            parent_row = {
                "id": parent_id,
                self.text_key: doc.page_content,
                "metadata": doc.metadata
            }
            parent_rows.append(parent_row)
            child_docs = self.child_splitter.split_documents([doc])
            for i, child_doc in enumerate(child_docs):
                child_id = self.client.generate_unique_hash()
                embedding = self.embedding_model.embed_query(
                    child_doc.page_content)
                child_row = {
                    "id": child_id,
                    "parent_id": parent_id,
                    self.text_key: child_doc.page_content,
                    "metadata": child_doc.metadata,
                    "embedding": embedding
                }
                child_rows.append(child_row)
        self.client.create_rows(self.parent_table_name,
                                parent_rows, self.dimension)
        self.client.create_rows(self.child_table_name,
                                child_rows, self.dimension)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant parent documents based on child embeddings."""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.client.search(
            self.child_table_name,
            query_embedding,
            top_k=self.search_kwargs.get("top_k", 10)
        )
        parent_ids = [result["parent_id"] for result in results]
        parent_rows = self.client.get_rows(
            self.parent_table_name,
            ids=parent_ids
        )
        return [
            Document(
                page_content=row[self.text_key],
                metadata=row["metadata"]
            ) for row in parent_rows
        ]

    def invoke(self, query: str) -> List[Document]:
        """Synchronous wrapper for retrieval."""
        return self._get_relevant_documents(query)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search using query text and return parent documents."""
        top_k = self.search_kwargs.get("top_k", k)
        query_embedding = self.embedding_model.embed_query(query)
        results = self.client.search(
            self.child_table_name,
            query_embedding,
            top_k=top_k,
            threshold=kwargs.get("threshold")
        )
        parent_ids = [result["parent_id"] for result in results]
        parent_rows = self.client.get_rows(
            self.parent_table_name,
            ids=parent_ids
        )
        return [
            Document(
                page_content=row[self.text_key],
                metadata=row["metadata"]
            ) for row in parent_rows
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding,
        metadatas: Optional[List[Dict]] = None,
        client: Optional[PgVectorClient] = None,
        database_name: str = "langchain",
        collection_name: str = "parent_doc",
        parent_table_name: str = "parent_doc_parent",
        child_table_name: str = "parent_doc_child",
        child_splitter=None,
        text_key: str = "page_content",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "PgVectorParentDocumentRetriever":
        """Initialize retriever from texts and embeddings."""
        if client is None:
            client = PgVectorClient(
                dbname=database_name,
                user="postgres",
                password="password",
                host="localhost",
                port=5432,
                overwrite_db=False
            )
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        retriever = cls(
            client=client,
            embedding_model=embedding,
            child_splitter=child_splitter,
            database_name=database_name,
            collection_name=collection_name,
            parent_table_name=parent_table_name,
            child_table_name=child_table_name,
            text_key=text_key,
            search_kwargs=search_kwargs
        )
        retriever.aadd_documents(documents)
        return retriever

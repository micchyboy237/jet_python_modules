import traceback
import chromadb
from chromadb import Settings, Documents
from chromadb.api.types import (
    OneOrMany,
    ID,
    Where,
    WhereDocument,
    Embedding,
    PyEmbedding,
    GetResult,
    QueryResult,
)
from chromadb.utils.batch_utils import create_batches

from typing import Callable, Optional, Union

from jet.db.chroma import (
    CHROMA_HTTP_HOST,
    CHROMA_HTTP_PORT,
    CHROMA_HTTP_HEADERS,
    CHROMA_HTTP_SSL,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    CHROMA_CLIENT_AUTH_PROVIDER,
    CHROMA_CLIENT_AUTH_CREDENTIALS,
)
from jet.db.chroma import VectorItem, InitialDataEntry, SearchResult, convert_search_results


class ChromaClient:
    def __init__(
        self,
            collection_name,
            embedding_function,
            *,
            data_path="data/vector_db",
            overwrite: bool = False,
            metadata: Optional[dict[str, any]] = None,
            initial_data: list[str] | list[InitialDataEntry] = [],
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        settings_dict = {
            "allow_reset": True,
            "anonymized_telemetry": False,
        }
        if CHROMA_CLIENT_AUTH_PROVIDER is not None:
            settings_dict["chroma_client_auth_provider"] = CHROMA_CLIENT_AUTH_PROVIDER
        if CHROMA_CLIENT_AUTH_CREDENTIALS is not None:
            settings_dict["chroma_client_auth_credentials"] = CHROMA_CLIENT_AUTH_CREDENTIALS
        if CHROMA_HTTP_HOST != "":
            self.client = chromadb.HttpClient(
                host=CHROMA_HTTP_HOST,
                port=CHROMA_HTTP_PORT,
                headers=CHROMA_HTTP_HEADERS,
                ssl=CHROMA_HTTP_SSL,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
                settings=Settings(**settings_dict),
            )
        else:
            self.client = chromadb.PersistentClient(
                path=data_path,
                settings=Settings(**settings_dict),
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )

        self.collection = self.select_collection(
            collection_name=collection_name,
            overwrite=overwrite,
            metadata=metadata,
            initial_data=initial_data,
            embedding_function=embedding_function
        )

    def select_collection(
        self,
        collection_name: str,
        overwrite: bool = False,
        metadata: Optional[dict[str, any]] = None,
        initial_data: list[str] | list[InitialDataEntry] = [],
        embedding_function: Callable = None,
    ):
        try:
            if self.has_collection(collection_name):
                if overwrite:
                    self.delete_collection(collection_name)

            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata,
                embedding_function=embedding_function,
            )

            if initial_data:
                # Extract ids and documents using zip
                ids = []
                documents: Documents = []
                embeddings = None  # Default as None
                metadatas = None  # Default as None

                # Check if initial_data is a list of strings
                if isinstance(initial_data, list) and all(isinstance(i, str) for i in initial_data):
                    # Convert each string into an entry with 'id' as index and 'document' as the string
                    initial_data = [{"id": str(index), "document": item}
                                    for index, item in enumerate(initial_data)]

                for item in initial_data:
                    if item.get("id"):
                        ids.append(item["id"])
                    if item.get("document"):
                        documents.append(item["document"])
                    if item.get("embeddings"):
                        if embeddings is None:
                            embeddings = []  # Initialize embeddings only when needed
                        # Append embeddings if present
                        embeddings.append(item["embeddings"])
                    if item.get("metadata"):
                        if metadatas is None:
                            metadatas = []  # Initialize metadatas only when needed
                        # Append metadatas if present
                        metadatas.append(item["metadata"])

                # If no embeddings were provided, generate them
                if embeddings is None:
                    embeddings = embedding_function(documents)

                # Add data to the collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

            return collection

        except Exception as e:
            from jet.logger import logger
            logger.error(
                f"Failed to create collection '{collection_name}': {e}")
            traceback.print_exc()
            raise e

    def has_collection(self, collection_name: str) -> bool:
        # Check if the collection exists based on the collection name.
        collections = self.client.list_collections()
        return collection_name in collections

    def delete_collection(self, collection_name: str):
        # Delete the collection based on the collection name.
        return self.client.delete_collection(name=collection_name)

    def query(
        self,
        *,
        texts: Optional[OneOrMany[str]] = None,
        embeddings: Optional[
            Union[OneOrMany[Embedding], OneOrMany[PyEmbedding]]
        ] = None,
        top_n: int = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: list = ["metadatas", "documents", "distances"],
    ) -> QueryResult:
        try:
            options = {
                "query_embeddings": embeddings,
                "query_texts": texts,
                "where": where,
                "where_document": where_document,
                "include": include,
            }
            if top_n:
                options["n_results"] = top_n
            return self.collection.query(**options)
        except Exception as e:
            from jet.logger import logger
            logger.error(
                f"Error running query on collection '{self.collection_name}': {e}")
            raise e

    def search(
        self,
        *,
        texts: Optional[OneOrMany[str]] = None,
        embeddings: Optional[
            Union[OneOrMany[Embedding], OneOrMany[PyEmbedding]]
        ] = None,
        top_n: int = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: list = ["metadatas", "documents", "distances"],
    ) -> list[SearchResult]:
        try:
            options = {
                "embeddings": embeddings,
                "texts": texts,
                "top_n": top_n,
                "where": where,
                "where_document": where_document,
                "include": include,
            }
            if not embeddings:
                embeddings = self.embedding_function(texts)

            query_result = self.query(**options)
            search_results = convert_search_results(query_result)
            return search_results
        except Exception as e:
            from jet.logger import logger
            logger.error(
                f"Error running query on collection '{self.collection_name}': {e}")
            raise e

    def get(
        self,
        *,
        ids: Optional[OneOrMany[ID]] = None,
        limit: int = None,
        offset: Optional[int] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: list = ["metadatas", "documents"],
    ) -> GetResult:
        try:
            options = {
                "ids": ids,
                "limit": limit,
                "offset": offset,
                "where": where,
                "where_document": where_document,
                "include": include,
            }
            return self.collection.get(**options)

        except Exception as e:
            from jet.logger import logger
            logger.error(
                f"Error running query on collection '{self.collection_name}': {e}")
            raise e

    def insert(self, items: list[VectorItem]):
        ids = [item["id"] for item in items]
        documents = [item["document"] for item in items]
        embeddings = [item["embeddings"] for item in items]
        metadatas = [item["metadata"] for item in items]
        for batch in create_batches(
            api=self.client,
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        ):
            self.collection.add(*batch)

    def upsert(self, items: list[VectorItem]):
        ids = [item["id"] for item in items if item.get("id")]
        documents = [item["document"]
                     for item in items if item.get("document")]
        embeddings = [item["embeddings"]
                      for item in items if item.get("embeddings")]
        metadatas = [item["metadata"]
                     for item in items if item.get("metadata")]

        # If no embeddings were provided, generate them
        if not embeddings:
            embeddings = self.embedding_function(documents)

        self.collection.upsert(ids=ids, documents=documents,
                               embeddings=embeddings, metadatas=metadatas)

    def delete(
        self,
        collection_name: str,
        ids: Optional[list[str]] = None,
        filter: Optional[dict] = None,
    ):
        # Delete the items from the collection based on the ids.
        collection = self.client.get_collection(name=collection_name)
        if collection:
            if ids:
                collection.delete(ids=ids)
            elif filter:
                collection.delete(where=filter)

    def reset(self):
        # Resets the database. This will delete all collections and item entries.
        return self.client.reset()

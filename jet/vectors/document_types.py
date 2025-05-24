from llama_index.core.schema import Document as BaseDocument, MetadataMode, TextNode
from typing import Any, List, Optional, TypedDict, cast, override
from pydantic import BaseModel, Field

# --- Pydantic Classes ---


class HeadersQueryResponse(BaseModel):
    data: List[str]


class RelevantDocument(BaseModel):
    document_number: int = Field(..., ge=0)
    confidence: int = Field(..., ge=1, le=10)


class DocumentSelectionResult(BaseModel):
    relevant_documents: List[RelevantDocument]
    evaluated_documents: List[int]
    feedback: str


class Document(BaseDocument):
    # @staticmethod
    # def rerank_documents(query: str | list[str], docs: list['Document'], model: str | OLLAMA_EMBED_MODELS | list[str] | list[OLLAMA_EMBED_MODELS] = "paraphrase-multilingual") -> list[SimilarityResult]:
    #     texts: list[str] = []
    #     ids: list[str] = []

    #     for doc in docs:
    #         # text = doc.text
    #         text = doc.get_recursive_text()

    #         texts.append(text)
    #         ids.append(doc.node_id)

    #     query_scores = query_similarity_scores(
    #         query, texts, model=model, ids=ids)
    #     texts = [result["text"] for result in query_scores]

    #     # Hybrid reranking
    #     # if isinstance(query, list):
    #     #     query_str = "\n".join(query)
    #     # else:
    #     #     query_str = query
    #     # bm25_results = bm25_plus_search(texts, query_str)

    #     # query_scores: list[SimilarityResult] = [
    #     #     {
    #     #         **query_scores[result["doc_index"]],
    #     #         "score": result["score"],
    #     #         "rank": rank
    #     #     }
    #     #     for rank, result in enumerate(bm25_results, 1)
    #     # ]

    #     return query_scores

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def get_recursive_text(self) -> str:
        """
        Get content of this node and all of its child nodes recursively.
        """
        texts = [self.text, "\n"]

        for child in self.child_nodes or []:
            texts.append(child.metadata["header"])

        if self.parent_node:
            texts.insert(0, self.parent_node.metadata["header"])

        return "\n".join(filter(None, texts))


class HeaderMetadata(TypedDict, total=False):
    doc_index: int
    header_level: int
    header: str
    parent_header: str | None
    content: str
    chunk_index: int | None
    token_count: int | None
    source_url: str | None


class HeaderDocument(Document):
    def __init__(self, **data: Any):
        super().__init__(text=data["text"])
        self.metadata_separator = "\n"
        # Initialize metadata with defaults and update with provided data
        default_metadata: HeaderMetadata = {
            "doc_index": data.get("doc_index", 0),
            "header_level": data.get("header_level", 0),
            "header": data.get("header", ""),
            "parent_header": data.get("parent_header", None),
            "content": data.get("content", ""),
            "chunk_index": data.get("chunk_index", None),
            "token_count": data.get("token_count", None),
            "source_url": data.get("source_url", None),
        }
        self.metadata = default_metadata  # type: ignore

    def __getitem__(self, key: str) -> Any:
        """Allow direct dictionary-like access to instance attributes or metadata."""
        if hasattr(self, key):
            return getattr(self, key)
        # Cast metadata to HeaderMetadata for type checker and IntelliSense
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def get_recursive_text(self) -> str:
        """
        Get content of this node and all of its child nodes recursively, using header property.
        """
        # Cast metadata to HeaderMetadata for type checker and IntelliSense
        metadata = cast(HeaderMetadata, self.metadata)
        texts = [self.text, "\n"]
        if metadata["parent_header"]:
            texts.insert(0, metadata["parent_header"])
        return "\n".join(filter(None, texts))


class HeaderTextNode(TextNode):
    text_template: str = Field(
        default="{parent_header}\n{header}\n\n{metadata_str}\n\n{content}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Initialize metadata with default values if not provided
        default_metadata: HeaderMetadata = {
            "doc_index": 0,
            "header_level": 0,
            "header": "",
            "parent_header": None,
            "content": "",
            "chunk_index": None,
            "token_count": None,
            "source_url": None,
        }
        # Update with provided metadata, ensuring type compatibility
        provided_metadata = kwargs.get("metadata", {})
        self.metadata = {**default_metadata, **
                         provided_metadata}  # type: ignore

    def __getitem__(self, key: str) -> Any:
        """Allow direct dictionary-like access to instance attributes or metadata."""
        if hasattr(self, key):
            return getattr(self, key)
        # Cast metadata to HeaderMetadata for type checker and IntelliSense
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        # Cast metadata to HeaderMetadata for type checker and IntelliSense
        metadata = cast(HeaderMetadata, self.metadata)
        content = self.text
        if self.text.startswith(metadata["header"]):
            content = "\n".join(self.text.splitlines()[1:])

        return self.text_template.format(
            parent_header=metadata["parent_header"] or "",
            header=metadata["header"],
            content=content,
            metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        # Cast metadata to HeaderMetadata for type checker and IntelliSense
        metadata = cast(HeaderMetadata, self.metadata)
        usable_metadata_keys = [
            "doc_index",
            "chunk_index",
        ]
        if mode == MetadataMode.EMBED:
            usable_metadata_keys = []
        metadata_str = self.metadata_separator.join(
            [
                self.metadata_template.format(
                    key=key, value=str(metadata[key]))
                for key in usable_metadata_keys
                if key in metadata and metadata[key] is not None
            ]
        )
        return metadata_str

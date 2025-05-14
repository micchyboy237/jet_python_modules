from llama_index.core.schema import Document as BaseDocument, MetadataMode
from typing import Any, List, Optional, override
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


class HeaderDocument(Document):
    doc_index: int = Field(default=0)
    header_level: int = Field(default=0)
    header: str = Field(default="")
    parent_header: Optional[str] = Field(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.text_template = "{metadata_str}\n\n{content}"
        self.metadata_separator = "\n"
        self.metadata.update({
            'doc_index': self.doc_index,
            'header_level': self.header_level,
            'header': self.header,
            'parent_header': self.parent_header,
        })

    def get_recursive_text(self) -> str:
        """
        Get content of this node and all of its child nodes recursively, using header property.
        """
        texts = [self.text, "\n"]

        if self.parent_header:
            texts.insert(0, self.parent_header)

        return "\n".join(filter(None, texts))

    @override
    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL):
        usable_metadata_keys = [
            "doc_index",
            "chunk_idx",
            "parent_header",
            "header",
        ]
        return self.metadata_separator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
                and value is not None
            ]
        )

import uuid
from llama_index.core.schema import (
    Document as BaseDocument,
    TextNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo
)
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, TypedDict, Union, cast, ItemsView


class HeaderMetadata(TypedDict, total=False):
    id: str
    doc_index: int
    header_level: int
    header: str
    parent_header: str
    content: str
    source_url: str
    chunk_index: int | None
    tokens: int | None
    links: List[str] | None
    texts: List[str] | None


class Match(TypedDict):
    word: str
    start_idx: int
    end_idx: int
    line: str


class Document(BaseDocument):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def get_recursive_text(self, node_registry: Optional[Dict[str, 'Document']] = None) -> str:
        texts = [self.text]
        parent_header = ""
        if node_registry and self.relationships.get(NodeRelationship.PARENT):
            parent_info = self.relationships[NodeRelationship.PARENT]
            if isinstance(parent_info, RelatedNodeInfo):
                parent_id = parent_info.node_id
                parent_node = node_registry.get(parent_id)
                if parent_node:
                    parent_header = parent_node.metadata.get("header", "")
        elif self.parent_node:
            parent_header = self.parent_node.metadata.get("header", "")
        child_headers = []
        if node_registry and self.relationships.get(NodeRelationship.CHILD):
            child_infos = self.relationships[NodeRelationship.CHILD]
            if isinstance(child_infos, list):
                for child_info in child_infos:
                    if isinstance(child_info, RelatedNodeInfo):
                        child_node = node_registry.get(child_info.node_id)
                        if child_node:
                            child_header = child_node.metadata.get(
                                "header", "")
                            if child_header:
                                child_headers.append(child_header)
        elif self.child_nodes:
            for child in self.child_nodes:
                child_header = child.metadata.get("header", "")
                if child_header:
                    child_headers.append(child_header)
        result_texts = [self.text]
        if parent_header or child_headers:
            result_texts.append("")
            if parent_header:
                result_texts.append(parent_header)
            result_texts.extend(child_headers)
            result = "\n".join(result_texts)
        else:
            result = self.text
        return result

    def items(self) -> ItemsView[str, Any]:
        """Return key-value pairs of attributes and metadata."""
        result = {}
        for attr in ["text", "metadata", "metadata_separator"]:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        return result.items()


class HeaderDocumentDict(TypedDict, total=False):
    text: str
    id: Optional[str]
    metadata: HeaderMetadata
    embedding: Optional[List[float]]


class HeaderDocument(Document):
    id: Optional[str] = Field(
        None, description="Unique identifier for the document")

    @staticmethod
    def from_list(documents: Union[List['HeaderDocument'], List[HeaderDocumentDict]]) -> List['HeaderDocument']:
        """
        Builds a list of HeaderDocument instances with parent-child relationships
        derived from header_level metadata using parent stack logic.

        Args:
            documents: List of HeaderDocument instances or HeaderDocumentDict dictionaries
                       with header_level metadata. Any relationships in the input are ignored.

        Returns:
            List of new HeaderDocument instances with derived relationships and parent_header metadata.
        """
        result: List[HeaderDocument] = []
        level_stack: List[tuple[int, HeaderDocument]] = []
        doc_map: Dict[str, HeaderDocument] = {}

        # Convert all inputs to HeaderDocument instances
        processed_docs: List[HeaderDocument] = []
        for doc in documents:
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                # Generate ID if not provided
                doc_id = doc.get("id", str(uuid.uuid4()))
                metadata["id"] = doc_id
                new_doc = HeaderDocument(
                    text=doc.get("text", ""),
                    id=doc_id,
                    metadata=metadata,
                    embedding=doc.get("embedding")
                )
            else:
                new_doc = HeaderDocument(
                    text=doc.text,
                    id=doc.id or str(uuid.uuid4()),
                    metadata=dict(doc.metadata),
                    embedding=doc.embedding
                )
            processed_docs.append(new_doc)

        for doc in processed_docs:
            metadata = dict(doc.metadata)
            header_level = metadata.get("header_level", 0)
            doc_id = cast(str, doc.id)

            # Clear any existing relationships to ensure derivation from stack
            doc.relationships = {}

            # Update parent relationships
            while level_stack and level_stack[-1][0] >= header_level:
                level_stack.pop()

            if level_stack:
                parent_level, parent_doc = level_stack[-1]
                doc.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_doc.id)
                doc.metadata["parent_header"] = parent_doc.metadata["header"]
                parent_children = parent_doc.relationships.get(
                    NodeRelationship.CHILD, [])
                if not isinstance(parent_children, list):
                    parent_children = [
                        parent_children] if parent_children else []
                parent_children.append(RelatedNodeInfo(node_id=doc_id))
                parent_doc.relationships[NodeRelationship.CHILD] = parent_children
            else:
                doc.metadata["parent_header"] = ""

            level_stack.append((header_level, doc))
            doc_map[doc_id] = doc
            result.append(doc)

        return result

    def __init__(self, **data: Any):
        data = data.copy()
        text: str = data.pop("text", "")
        super().__init__(text=text, **data)
        metadata = data.pop("metadata", None)
        default_values: HeaderMetadata = {
            "doc_index": 0,
            "header_level": 0,
            "header": "",
            "parent_header": "",
            "content": "",
            "chunk_index": None,
            "tokens": None,
            # Prioritize source_url from data
            "source_url": data.get("source_url", None),
            "links": None,
            "texts": text.splitlines(),
        }
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        default_metadata = {
            **default_values,
            **metadata_dict,  # metadata_dict overrides defaults
            # Other data fields, excluding source_url
            **{k: v for k, v in data.items() if k in default_values and k != "source_url"},
        }
        if "source_url" in metadata_dict:  # Explicitly check metadata for source_url
            default_metadata["source_url"] = metadata_dict["source_url"]
        id = data.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id
        self.node_id = id
        self.metadata = HeaderMetadata(**default_metadata)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def __iter__(self):
        for attr in ["text", "metadata", "metadata_separator"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata.get(key, default)

    def get_recursive_text(self) -> str:
        metadata = cast(HeaderMetadata, self.metadata)
        texts = []
        if metadata["parent_header"]:
            texts.append(metadata["parent_header"])
        texts.append(self.text)
        result = "\n".join(filter(None, texts))
        if result:
            result += "\n"
        return result

    def items(self) -> ItemsView[str, Any]:
        """Return key-value pairs of attributes and metadata."""
        result = {}
        for attr in ["text", "metadata", "metadata_separator"]:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        return result.items()


class HeaderTextNode(TextNode):
    id: Optional[str] = Field(
        None, description="Unique identifier for the text node")
    text_template: str = Field(
        default="{parent_header}\n{header}\n\n{metadata_str}\n\n{content}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs = kwargs.copy()
        super().__init__(*args, **kwargs)
        default_metadata: HeaderMetadata = {
            "doc_index": 0,
            "header_level": 0,
            "header": "",
            "parent_header": "",
            "content": "",
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "texts": None,
        }
        provided_metadata = kwargs.get("metadata", {})
        id = kwargs.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id
        self.node_id = id
        self.metadata = {**default_metadata, **provided_metadata, "id": id}

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def __iter__(self):
        for attr in ["text", "metadata", "metadata_template", "metadata_separator", "text_template"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata.get(key, default)

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        metadata = cast(HeaderMetadata, self.metadata)
        content = self.text
        if metadata_mode != MetadataMode.NONE and self.text.startswith(metadata["header"]):
            content = "\n".join(self.text.splitlines()[1:]).strip()
        if metadata_mode == MetadataMode.NONE:
            result = content
        else:
            metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
            result = self.text_template.format(
                parent_header=metadata["parent_header"],
                header=metadata["header"],
                content=content,
                metadata_str=metadata_str
            ).strip()
        return result

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        metadata = cast(HeaderMetadata, self.metadata)
        usable_metadata_keys = ["id", "doc_index", "chunk_index"] if mode == MetadataMode.ALL else [
            "parent_header", "header"]
        metadata_str = self.metadata_separator.join(
            [
                self.metadata_template.format(
                    key=key, value=str(metadata[key]))
                for key in usable_metadata_keys
                if key in metadata and metadata[key] is not None and str(metadata[key]).strip()
            ]
        )
        return metadata_str

    def items(self) -> ItemsView[str, Any]:
        """Return key-value pairs of attributes and metadata."""
        result = {}
        for attr in ["text", "metadata", "metadata_template", "metadata_separator", "text_template"]:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        return result.items()


class HeaderDocumentWithScore(BaseModel):
    """A class that wraps a HeaderDocument with a score, similar to NodeWithScore, and includes SearchResult properties."""
    node: HeaderDocument
    score: Optional[float] = None
    doc_index: int = Field(
        default=0, description="Index of the document in the original list")
    rank: int = Field(
        default=0, description="Rank of the document in search results")
    combined_score: float = Field(
        default=0.0, description="Combined BM25 and embedding score")
    embedding_score: float = Field(
        default=0.0, description="Embedding-based score")
    headers: List[str] = Field(
        default_factory=list, description="List of headers associated with the chunk")
    highlighted_text: str = Field(
        default="", description="Text with query terms highlighted")
    matches: List[Match] = Field(
        default_factory=list, description="List of query term matches")

    def __init__(self, **data: Any):
        super().__init__(**data)

    def __str__(self) -> str:
        score_str = "None" if self.score is None else f"{self.score:0.3f}"
        result = f"{self.node}\nScore: {score_str}\nRank: {self.rank}\nCombined Score: {self.combined_score:0.3f}\nEmbedding Score: {self.embedding_score:0.3f}"
        return result

    def get_score(self, raise_error: bool = False) -> float:
        if self.score is None:
            if raise_error:
                raise ValueError("Score not set.")
            return 0.0
        return self.score

    @classmethod
    def class_name(cls) -> str:
        return "HeaderDocumentWithScore"

    @property
    def id(self) -> str:
        """Unique identifier for the document, alias for id_."""
        return self.node.id

    @property
    def node_id(self) -> str:
        return self.node.node_id

    @property
    def id_(self) -> str:
        return self.node.id_

    @property
    def text(self) -> str:
        return self.node.text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.node.metadata

    @property
    def embedding(self) -> Optional[List[float]]:
        return self.node.embedding

    def get_text(self) -> str:
        return self.node.get_text()

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        return self.node.get_content(metadata_mode=metadata_mode)

    def get_embedding(self) -> List[float]:
        return self.node.get_embedding()

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.node[key]

    def __iter__(self):
        for attr in ["node", "score", "doc_index", "rank", "combined_score", "embedding_score", "headers", "highlighted_text", "matches"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)
        for key, value in self.node:
            if value is not None:
                yield key, value

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.node.get(key, default)

    def items(self) -> ItemsView[str, Any]:
        """Return key-value pairs of attributes and node properties."""
        result = {}
        for attr in ["node", "score", "doc_index", "rank", "combined_score", "embedding_score", "headers", "highlighted_text", "matches"]:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        return result.items()

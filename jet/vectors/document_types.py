from pathlib import Path
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

from jet.code.markdown_utils import parse_markdown


class DotDict(dict):
    """A dictionary subclass that supports dot notation access."""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'")


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
    texts: List['HeaderTextNode'] | None


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
    id: str = Field(..., description="Unique identifier for the document")

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
        processed_docs: List[HeaderDocument] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                doc_id = doc.get("id", str(uuid.uuid4()))
                metadata["id"] = doc_id
                # Ensure doc_index and chunk_index are set
                if "doc_index" not in metadata or metadata["doc_index"] is None:
                    metadata["doc_index"] = idx
                if "chunk_index" not in metadata or metadata["chunk_index"] is None:
                    metadata["chunk_index"] = 0
                new_doc = HeaderDocument(
                    text=doc.get("text", ""),
                    id=doc_id,
                    metadata=metadata,
                    embedding=doc.get("embedding")
                )
            else:
                # doc is HeaderDocument
                doc_metadata = dict(doc.metadata)
                if "doc_index" not in doc_metadata or doc_metadata["doc_index"] is None:
                    doc_metadata["doc_index"] = idx
                if "chunk_index" not in doc_metadata or doc_metadata["chunk_index"] is None:
                    doc_metadata["chunk_index"] = 0
                new_doc = HeaderDocument(
                    text=doc.text,
                    id=doc.id or str(uuid.uuid4()),
                    metadata=doc_metadata,
                    embedding=doc.embedding
                )
            processed_docs.append(new_doc)
        for doc in processed_docs:
            metadata = DotDict(doc.metadata)
            header_level = metadata.get("header_level", 0)
            doc_id = cast(str, doc.id)
            doc.relationships = {}
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

    @staticmethod
    def from_markdown(md_input: Union[str, Path], source_url: Optional[str] = None) -> List['HeaderDocument']:
        """Convert markdown content into a list of HeaderDocument instances."""
        tokens = parse_markdown(md_input)
        documents: List[HeaderDocumentDict] = []
        current_content: List[str] = []
        current_header: str = ""
        current_level: int = 0
        doc_index: int = 0

        for token in tokens:
            if token['type'] == 'header' and token['level'] is not None:
                if current_content or current_header:
                    # Save the previous document if it exists
                    documents.append({
                        'text': '\n'.join(current_content).strip(),
                        'metadata': {
                            'header': current_header,
                            'header_level': current_level,
                            'doc_index': doc_index,
                            'content': '\n'.join(current_content).strip(),
                            'source_url': source_url,
                            'chunk_index': 0,
                            'texts': [HeaderTextNode(text=line, id=str(uuid.uuid4())) for line in current_content]
                        }
                    })
                    doc_index += 1
                    current_content = []
                current_header = token['content'].lstrip('#').strip()
                current_level = token['level']
            else:
                current_content.append(token['content'])

        # Save the last document if it exists
        if current_content or current_header:
            documents.append({
                'text': '\n'.join(current_content).strip(),
                'metadata': {
                    'header': current_header,
                    'header_level': current_level,
                    'doc_index': doc_index,
                    'content': '\n'.join(current_content).strip(),
                    'source_url': source_url,
                    'chunk_index': 0,
                    'texts': [HeaderTextNode(text=line, id=str(uuid.uuid4())) for line in current_content]
                }
            })

        return HeaderDocument.from_list(documents)

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
            "source_url": data.get("source_url", None),
            "links": None,
            "texts": [HeaderTextNode(text=line, id=str(uuid.uuid4())) for line in text.splitlines()],
        }
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        default_metadata = DotDict({
            **default_values,
            **metadata_dict,
            **{k: v for k, v in data.items() if k in default_values and k != "source_url"},
        })
        if "source_url" in metadata_dict:
            default_metadata["source_url"] = metadata_dict["source_url"]
        id = data.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id
        self.node_id = id
        self.metadata = default_metadata

    def __getattr__(self, key: str) -> Any:
        if key in self.metadata:
            return self.metadata[key]
        return super().__getattr__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in HeaderMetadata.__annotations__:
            self.metadata[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]

    def __iter__(self):
        for attr in ["text", "metadata", "metadata_separator"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def get_recursive_text(self) -> str:
        texts = []
        if self.metadata.parent_header:
            texts.append(self.metadata.parent_header)
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
        default="{parent_header}\n{header}\n\n{metadata_str}\n{content}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs = kwargs.copy()
        super().__init__(*args, **kwargs)
        default_metadata: HeaderMetadata = {
            "doc_index": "",
            "header_level": 0,
            "header": "",
            "parent_header": "",
            "content": "",
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "texts": None,
            "id": "",
        }
        provided_metadata = kwargs.get("metadata", {})
        id = kwargs.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id
        self.node_id = id
        self.metadata = DotDict(
            {**default_metadata, **provided_metadata, "id": id})

    def __str__(self) -> str:
        """Return the text content as a string for compatibility with existing code."""
        return self.text

    def __getattr__(self, key: str) -> Any:
        if key in self.metadata:
            return self.metadata[key]
        return super().__getattr__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in HeaderMetadata.__annotations__:
            self.metadata[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]

    def __iter__(self):
        for attr in ["text", "metadata", "metadata_template", "metadata_separator", "text_template"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata.get(key, default)

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        content = self.text
        if metadata_mode != MetadataMode.NONE and self.text.startswith(self.metadata.header):
            content = "\n".join(self.text.splitlines()[1:]).strip()
        if metadata_mode == MetadataMode.NONE:
            result = content
        else:
            metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
            result = self.text_template.format(
                parent_header=self.metadata.parent_header,
                header=self.metadata.header,
                content=content,
                metadata_str=metadata_str
            ).strip()
        return result

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        usable_metadata_keys = ["id", "doc_index", "chunk_index"] if mode == MetadataMode.ALL else [
            "parent_header", "header"]
        metadata_str = self.metadata_separator.join(
            [
                self.metadata_template.format(
                    key=key, value=str(self.metadata[key]))
                for key in usable_metadata_keys
                if key in self.metadata and self.metadata[key] is not None and str(self.metadata[key]).strip()
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

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        return getattr(self.node, key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self.__fields__ or key in HeaderMetadata.__annotations__:
            if key in HeaderMetadata.__annotations__:
                self.node.metadata[key] = value
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

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

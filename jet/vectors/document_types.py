from llama_index.core.schema import Document, TextNode, MetadataMode
from pydantic import Field
from typing import Any, List, TypedDict, cast
from llama_index.core.schema import Document as BaseDocument, MetadataMode, TextNode
from typing import Any, List, Optional, TypedDict, cast
from pydantic import BaseModel, Field


class Document(BaseDocument):
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


class HeaderDocument(Document):
    id: Optional[str] = Field(None,
                              description="Unique identifier for the document")

    def __init__(self, **data: Any):
        # Make a defensive copy to avoid modifying input
        data = data.copy()
        # Pop for parent initialization
        text: str = data.pop("text", "")
        # Pass all remaining data to parent, including id
        super().__init__(text=text, **data)

        # Pop metadata after parent initialization
        metadata = data.pop("metadata", None)

        # Define default metadata values
        default_values: HeaderMetadata = {
            "doc_index": 0,
            "header_level": 0,
            "header": "",
            "parent_header": "",
            "content": "",
            "chunk_index": None,
            "tokens": None,
            "source_url": None,
            "links": None,
            "texts": text.splitlines(),
        }

        # Merge metadata (if valid) with defaults, prioritizing metadata
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        default_metadata = {
            **default_values,
            **data,  # Include any extra keys from data
            **metadata_dict,  # Metadata overrides defaults and data
        }

        # Ensure metadata["id"] matches id
        id = data.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id  # Explicit for clarity
        self.node_id = id
        self.metadata = HeaderMetadata(**default_metadata)

    def __getitem__(self, key: str) -> Any:
        """Allow direct dictionary-like access to instance attributes or metadata."""
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def __iter__(self):
        """Enable **obj unpacking by yielding key-value pairs for attributes and metadata."""
        for attr in ["text", "metadata", "metadata_separator"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)
        metadata = cast(HeaderMetadata, self.metadata)
        for key, value in metadata.items():
            if value is not None:  # Only include non-None metadata
                yield key, value

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute or metadata value by key, returning default if not found."""
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata.get(key, default)

    def get_recursive_text(self) -> str:
        """
        Get content of this node and all of its child nodes recursively, using header property.
        """
        metadata = cast(HeaderMetadata, self.metadata)
        texts = [self.text, "\n"]
        if metadata["parent_header"]:
            texts.insert(0, metadata["parent_header"])
        return "\n".join(filter(None, texts))


class HeaderTextNode(TextNode):
    id: Optional[str] = Field(None,
                              description="Unique identifier for the text node")
    text_template: str = Field(
        default="{parent_header}\n{header}\n\n{metadata_str}\n\n{content}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Defensive copy of kwargs
        kwargs = kwargs.copy()
        # Pass all args and kwargs to parent, including id
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
        # Ensure metadata["id"] matches self.id_
        id = kwargs.pop("id", self.id_)
        default_metadata["id"] = id
        self.id = id  # Explicit for clarity
        self.node_id = id
        self.metadata = {**default_metadata, **
                         provided_metadata, "id": id}

    def __getitem__(self, key: str) -> Any:
        """Allow direct dictionary-like access to instance attributes or metadata."""
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata[key]

    def __iter__(self):
        """Enable **obj unpacking by yielding key-value pairs for attributes and metadata."""
        for attr in ["text", "metadata", "metadata_template", "metadata_separator", "text_template"]:
            if hasattr(self, attr):
                yield attr, getattr(self, attr)
        metadata = cast(HeaderMetadata, self.metadata)
        for key, value in metadata.items():
            if value is not None:  # Only include non-None metadata
                yield key, value

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute or metadata value by key, returning default if not found."""
        if hasattr(self, key):
            return getattr(self, key)
        metadata = cast(HeaderMetadata, self.metadata)
        return metadata.get(key, default)

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text
        metadata = cast(HeaderMetadata, self.metadata)
        content = self.text
        if self.text.startswith(metadata["header"]):
            content = "\n".join(self.text.splitlines()[1:])
        return self.text_template.format(
            parent_header=metadata["parent_header"],
            header=metadata["header"],
            content=content,
            metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        metadata = cast(HeaderMetadata, self.metadata)
        usable_metadata_keys = ["id", "doc_index", "chunk_index"]
        if mode == MetadataMode.EMBED:
            usable_metadata_keys = ["parent_header", "header"]
        metadata_str = self.metadata_separator.join(
            [
                self.metadata_template.format(
                    key=key, value=str(metadata[key]))
                for key in usable_metadata_keys
                if key in metadata and metadata[key] is not None
            ]
        )
        return metadata_str

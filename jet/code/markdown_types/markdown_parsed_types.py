from typing import Literal, TypedDict, Union


class ListItem(TypedDict, total=False):
    text: str
    task_item: bool
    checked: bool | None


class CodeMeta(TypedDict, total=False):
    language: str | None
    code_type: Literal["indented"] | None


class TableMeta(TypedDict):
    header: list[str]
    rows: list[list[str]]


class ListMeta(TypedDict):
    items: list[ListItem]


MetaType = Union[ListMeta, CodeMeta, TableMeta, dict]

ContentType = Literal[
    "header",
    "paragraph",
    "blockquote",
    "code",
    "table",
    "unordered_list",
    "ordered_list",
    "html_block",
    # Custom types
    "head",
    "json",
]


class MarkdownToken(TypedDict):
    type: ContentType
    content: str
    level: int | None
    meta: MetaType | None
    line: int


class HeaderDoc(TypedDict):
    id: str
    doc_index: int
    header: str
    content: str
    level: int | None
    parent_headers: list[str]
    parent_header: str | None
    parent_level: int | None
    source: str | None
    tokens: list[MarkdownToken]


class HeaderSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""

    doc_index: int
    doc_id: str
    level: int | None
    parent_level: int | None
    parent_headers: list[str]
    start_idx: int
    end_idx: int
    chunk_idx: int
    source: str
    header_content_similarity: float
    headers_similarity: float
    content_similarity: float
    num_tokens: int
    preprocessed_header: str
    preprocessed_headers_context: str
    preprocessed_content: str


class HeaderSearchResult(TypedDict):
    """Typed dictionary for search result structure."""

    id: str
    rank: int
    score: float
    header: str
    parent_header: str | None
    content: str
    metadata: HeaderSearchMetadata


__all__ = [
    "ListItem",
    "CodeMeta",
    "TableMeta",
    "ListMeta",
    "MetaType",
    "ContentType",
    "MarkdownToken",
    "HeaderDoc",
    "HeaderSearchMetadata",
    "HeaderSearchResult",
]

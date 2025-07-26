from typing import TypedDict, List, Optional, Literal, Union


class ListItem(TypedDict, total=False):
    text: str
    task_item: bool
    checked: Optional[bool]


class CodeMeta(TypedDict, total=False):
    language: Optional[str]
    code_type: Optional[Literal["indented"]]


class TableMeta(TypedDict):
    header: List[str]
    rows: List[List[str]]


class ListMeta(TypedDict):
    items: List[ListItem]


MetaType = Union[ListMeta, CodeMeta, TableMeta, dict]

ContentType = Literal[
    "header",
    "paragraph",
    "blockquote",
    "code",
    "table",
    "unordered_list",
    "ordered_list",
    "html_block"
]


class MarkdownToken(TypedDict):
    type: ContentType
    content: str
    level: Optional[int]
    meta: Optional[MetaType]
    line: int


class HeaderDoc(TypedDict):
    doc_index: int
    doc_id: str
    header: str
    content: str
    level: Optional[int]
    parent_headers: List[str]
    parent_header: Optional[str]
    parent_level: Optional[int]
    tokens: List[MarkdownToken]


class HeaderSearchMetadata(TypedDict):
    """Typed dictionary for search result metadata."""
    doc_index: int
    doc_id: str
    level: Optional[int]
    parent_level: Optional[int]
    start_idx: int
    end_idx: int
    chunk_idx: int
    header_content_similarity: float
    headers_similarity: float
    content_similarity: float
    num_tokens: int
    preprocessed_header: str
    preprocessed_headers_context: str
    preprocessed_content: str


class HeaderSearchResult(TypedDict):
    """Typed dictionary for search result structure."""
    rank: int
    score: float
    header: str
    parent_header: Optional[str]
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

from typing import TypedDict, List, Optional, Literal, Union

# For parse_markdown


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
    meta: Optional[MetaType]  # Allow None for meta
    line: int


__all__ = [
    "ListItem",
    "CodeMeta",
    "TableMeta",
    "ListMeta",
    "MetaType",
    "ContentType",
    "MarkdownToken",
]

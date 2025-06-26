from pathlib import Path
from typing import List, Optional, Union, Literal, TypedDict, Dict, Any
from pydantic import BaseModel, Field
import uuid


# Type definitions for MarkdownToken and meta data


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


class MarkdownToken(TypedDict):
    type: Literal[
        "header",
        "paragraph",
        "blockquote",
        "code",
        "table",
        "unordered_list",
        "ordered_list",
        "html_block"
    ]
    content: str
    level: Optional[int]
    meta: MetaType
    line: int


class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    line: int

    class Config:
        arbitrary_types_allowed = True


class TextNode(Node):
    type: str
    content: str
    meta: MetaType


class HeaderNode(Node):
    title: str
    level: int
    children: List[Union['HeaderNode', TextNode]] = Field(default_factory=list)

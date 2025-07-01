from typing import List, Optional, Union
from pydantic import BaseModel, Field
from jet.code.markdown_types import MarkdownToken, ContentType, MetaType
import uuid

# Base Node class for shared attributes


class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"auto_{uuid.uuid4().hex[:8]}")
    line: int
    parent_id: Optional[str] = None
    parent_header: Optional[str] = None
    type: ContentType
    header: str
    content: str
    chunk_index: int = 0
    _parent_node: Optional['NodeType'] = None

    def get_parent_node(self) -> Optional['NodeType']:
        """Returns the parent node of the current node."""
        return self._parent_node

    class Config:
        arbitrary_types_allowed = True

# Text Node for non-header content


class TextNode(Node):
    # Default to empty dict, allow None
    meta: Optional[MetaType] = Field(default_factory=dict)

# Header Node for header content


class HeaderNode(Node):
    type: ContentType = "header"
    level: int
    children: List['NodeType'] = Field(default_factory=list)


NodeType = Union[HeaderNode, TextNode]
Nodes = List[NodeType]

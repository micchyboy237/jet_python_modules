from typing import List, Optional, Union
from pydantic import BaseModel, Field
from jet.code.markdown_types import MarkdownToken, ContentType, MetaType
import uuid
from jet.logger import logger


class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"auto_{uuid.uuid4().hex[:8]}")
    doc_id: str  # Changed to required field
    line: int
    parent_id: Optional[str] = None
    parent_header: Optional[str] = None
    type: ContentType
    header: str
    content: str
    chunk_index: int = 0
    num_tokens: Optional[int] = 0
    _parent_node: Optional['NodeType'] = None

    def get_parent_node(self) -> Optional['NodeType']:
        """Returns the parent node of the current node."""
        return self._parent_node

    def get_text(self) -> str:
        """Combines header (if exists) and content with a newline separator."""
        parts = []
        if self.header:
            parts.append(self.header)
        if self.content:
            parts.append(self.content)
        return "\n".join(parts)

    def get_parent_headers(self) -> List[str]:
        """Returns a list of parent headers up to the root."""
        headers = []
        current = self
        seen_ids = set()
        logger.debug(f"Starting get_parent_headers for node {self.id}")
        while current.parent_header:
            if current.id in seen_ids:
                logger.error(
                    f"Cycle detected in node {current.id} with parent_header {current.parent_header}")
                break
            seen_ids.add(current.id)
            headers.append(current.parent_header)
            logger.debug(
                f"Processing node {current.id}, parent_header: {current.parent_header}, parent_node: {current._parent_node.id if current._parent_node else None}")
            next_node = current.get_parent_node()
            if not next_node:
                logger.debug(
                    f"No parent node for {current.id}, stopping traversal")
                break
            current = next_node
        logger.debug(
            f"Finished get_parent_headers for node {self.id}, headers: {headers}")
        return headers[::-1]

    class Config:
        arbitrary_types_allowed = True


class TextNode(Node):
    meta: Optional[MetaType] = Field(default_factory=dict)


class HeaderNode(Node):
    type: ContentType = "header"
    level: int
    children: List['NodeType'] = Field(default_factory=list)

    def get_recursive_text(self) -> str:
        """Recursively joins texts from current and children nodes with double newlines."""
        texts = [self.get_text()]
        for child in self.children:
            if isinstance(child, HeaderNode):
                texts.append(child.get_recursive_text())
            else:
                texts.append(child.get_text())
        return "\n\n".join(text for text in texts if text)


NodeType = Union[HeaderNode, TextNode]
Nodes = List[NodeType]

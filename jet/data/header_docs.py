from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from jet.code.markdown_types import (
    MarkdownToken,
    ListItem,
    ListMeta,
    CodeMeta,
    TableMeta,
    MetaType,
    ContentType
)
import uuid

from jet.code.markdown_utils import derive_text, parse_markdown

# Type definitions for MarkdownToken and meta data


# Base Node class for shared attributes


class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"auto_{uuid.uuid4().hex[:8]}")
    parent_id: Optional[str] = None
    line: int

    class Config:
        arbitrary_types_allowed = True

# Text Node for non-header content


class TextNode(Node):
    type: ContentType
    content: str
    meta: MetaType

# Header Node for header content


class HeaderNode(Node):
    type: ContentType = "header"
    content: str
    level: int
    children: List[Union['HeaderNode', TextNode]] = Field(default_factory=list)


Nodes = List[Union[HeaderNode, TextNode]]

# Header Tree to manage the hierarchy


class HeaderDocs(BaseModel):
    root: Nodes = Field(default_factory=list)

    @staticmethod
    def from_tokens(tokens: List[MarkdownToken]) -> 'HeaderDocs':
        """
        Converts a list of MarkdownToken into a HeaderDocs with parent-child relationships.
        """
        root: Nodes = []
        parent_stack: List[HeaderNode] = []
        seen_ids: set = set()

        def generate_unique_id() -> str:
            new_id = f"auto_{uuid.uuid4().hex[:8]}"
            while new_id in seen_ids:
                new_id = f"auto_{uuid.uuid4().hex[:8]}"
            seen_ids.add(new_id)
            return new_id

        for token in tokens:
            if token['type'] == 'header' and token['level'] is not None:
                header_id = generate_unique_id()
                parent_id = parent_stack[-1].id if parent_stack else None
                new_header = HeaderNode(
                    content=token['content'],
                    level=token['level'],
                    line=token['line'],
                    parent_id=parent_id,
                    id=header_id
                )

                while parent_stack and parent_stack[-1].level >= new_header.level:
                    parent_stack.pop()

                if parent_stack:
                    parent_stack[-1].children.append(new_header)
                else:
                    new_header.parent_id = None
                    root.append(new_header)

                parent_stack.append(new_header)
            else:
                text_id = generate_unique_id()
                text_node = TextNode(
                    type=token['type'],
                    content=derive_text(token),
                    meta=token['meta'],
                    line=token['line'],
                    parent_id=parent_stack[-1].id if parent_stack else None,
                    id=text_id
                )

                if parent_stack:
                    parent_stack[-1].children.append(text_node)
                else:
                    root.append(text_node)

        return HeaderDocs(root=root)

    @staticmethod
    def from_string(input: Union[str, Path]) -> 'HeaderDocs':
        tokens = parse_markdown(input)
        return HeaderDocs.from_tokens(tokens)

    def as_texts(self) -> List[str]:
        """
        Returns a list of Markdown text representations for all nodes in the tree.
        """
        texts: List[str] = []

        def traverse(node: Union[HeaderNode, TextNode]) -> None:
            if isinstance(node, HeaderNode):
                texts.append(
                    f"{'#' * node.level} {node.content.strip()}" if node.content else "")
                for child in node.children:
                    traverse(child)
            else:
                texts.append(node.content)

        for node in self.root:
            traverse(node)
        return texts

    def as_nodes(self) -> Nodes:
        """
        Returns a flattened list of all nodes in document order with hashtags prepended to header content.
        """
        nodes: Nodes = []

        def traverse(node: Union[HeaderNode, TextNode]) -> None:
            if isinstance(node, HeaderNode):
                # Create a new HeaderNode with hashtags prepended to content
                modified_node = HeaderNode(
                    id=node.id,
                    parent_id=node.parent_id,
                    line=node.line,
                    content=f"{node.content.strip()}" if node.content else "",
                    level=node.level,
                    children=node.children
                )
                nodes.append(modified_node)
                for child in node.children:
                    traverse(child)
            else:
                nodes.append(node)

        for node in self.root:
            traverse(node)
        return nodes

    def as_tree(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the tree structure.
        """
        def node_to_dict(node: Union[HeaderNode, TextNode]) -> Dict[str, Any]:
            base = {
                "id": node.id,
                "parent_id": node.parent_id,
                "line": node.line,
            }
            if isinstance(node, HeaderNode):
                base.update({
                    "type": "header",
                    "content": node.content,
                    "level": node.level,
                    "children": [node_to_dict(child) for child in node.children]
                })
            else:
                base.update({
                    "type": node.type,
                    "content": node.content,
                    "meta": node.meta
                })
            return base

        return {
            "root": [node_to_dict(node) for node in self.root]
        }

from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from jet.code.markdown_types import (
    MarkdownToken,
)
from jet.code.markdown_utils import derive_text, parse_markdown
import uuid

from jet.data.header_types import Nodes, TextNode, HeaderNode


# Header Tree to manage the hierarchy
class HeaderDocs(BaseModel):
    root: Nodes = Field(default_factory=list)
    tokens: List[MarkdownToken] = Field(default_factory=list)

    @staticmethod
    def from_tokens(tokens: List[MarkdownToken]) -> 'HeaderDocs':
        root: Nodes = []
        parent_stack: List[HeaderNode] = []
        seen_ids: set = set()
        id_to_node: Dict[str, Union[HeaderNode, TextNode]] = {}

        def generate_unique_id() -> str:
            new_id = str(uuid.uuid4())
            while new_id in seen_ids:
                new_id = str(uuid.uuid4())
            seen_ids.add(new_id)
            return new_id

        for token in tokens:
            if token['type'] == 'header' and token['level'] is not None:
                header_id = generate_unique_id()
                text = derive_text(token)
                header = text.splitlines()[0].strip()
                content = "\n".join(text.splitlines()[1:]).strip()
                new_header = HeaderNode(
                    header=header,
                    content=content,
                    level=token['level'],
                    line=token['line'],
                    id=header_id,
                )
                id_to_node[header_id] = new_header
                while parent_stack and parent_stack[-1].level >= new_header.level:
                    parent_stack.pop()
                if parent_stack:
                    new_header.parent_id = parent_stack[-1].id
                    new_header._parent_node = id_to_node[new_header.parent_id]
                    new_header.parent_header = new_header._parent_node.header
                    parent_stack[-1].children.append(new_header)
                else:
                    root.append(new_header)
                parent_stack.append(new_header)
            else:
                text_id = generate_unique_id()
                text = derive_text(token)
                header = text.splitlines()[0].strip()
                content = "\n".join(text.splitlines()[1:]).strip()
                # Handle None meta by converting to empty dict
                meta = token['meta'] if token['meta'] is not None else {}
                text_node = TextNode(
                    type=token['type'],
                    header=header,
                    content=content,
                    meta=meta,
                    line=token['line'],
                    parent_id=parent_stack[-1].id if parent_stack else None,
                    id=text_id
                )
                id_to_node[text_id] = text_node
                if text_node.parent_id and text_node.parent_id in id_to_node:
                    text_node._parent_node = id_to_node[text_node.parent_id]
                    text_node.parent_header = text_node._parent_node.header
                if parent_stack:
                    parent_stack[-1].children.append(text_node)
                else:
                    root.append(text_node)
        return HeaderDocs(root=root, tokens=tokens)

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
                texts.append(f"{node.content.strip()}" if node.content else "")
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
                    header=node.header,
                    content=node.content,
                    id=node.id,
                    parent_id=node.parent_id,
                    parent_header=node.parent_header,
                    line=node.line,
                    level=node.level,
                    children=node.children,
                    _parent_node=node._parent_node
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

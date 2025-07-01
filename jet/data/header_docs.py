from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from jet.code.markdown_types import (
    MarkdownToken,
)
from jet.code.markdown_utils import derive_text, parse_markdown
import uuid
from jet.data.header_types import Nodes, TextNode, HeaderNode
from jet.logger import logger


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

        logger.debug(f"Processing {len(tokens)} tokens")
        for token in tokens:
            logger.debug(
                f"Processing token type: {token['type']}, content: {token['content'][:50]}...")
            if token['type'] == 'header' and token['level'] is not None:
                header_id = generate_unique_id()
                text = derive_text(token)
                header_lines = text.splitlines()
                header = header_lines[0].lstrip('#').strip()
                content = "\n".join(header_lines[1:]).strip()
                logger.debug(
                    f"Creating HeaderNode: header={header}, content={content}, level={token['level']}")
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
                    logger.debug(
                        f"Added header {header_id} as child of {parent_stack[-1].id}")
                else:
                    root.append(new_header)
                    logger.debug(f"Added header {header_id} to root")
                parent_stack.append(new_header)
            else:
                text_id = generate_unique_id()
                text = derive_text(token)
                header_lines = text.splitlines()
                header = header_lines[0].strip(
                ) if header_lines else text.strip()
                content = text.strip()
                meta = token['meta'] if token['meta'] is not None else {}
                logger.debug(
                    f"Creating TextNode: header={header}, content={content}, type={token['type']}")
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
                    logger.debug(
                        f"Set parent for text node {text_id} to {text_node.parent_id}")
                if parent_stack:
                    parent_stack[-1].children.append(text_node)
                    logger.debug(
                        f"Added text node {text_id} as child of {parent_stack[-1].id}")
                else:
                    root.append(text_node)
                    logger.debug(f"Added text node {text_id} to root")
        logger.debug(f"Finished processing tokens, root nodes: {len(root)}")
        return HeaderDocs(root=root, tokens=tokens)

    @staticmethod
    def from_string(input: Union[str, Path]) -> 'HeaderDocs':
        logger.debug(
            f"Parsing input of type {type(input).__name__}, content: {str(input)[:100]}...")
        tokens = parse_markdown(input)
        logger.debug(
            f"Parsed {len(tokens)} tokens: {[t['content'][:50] + '...' for t in tokens]}")
        if not tokens:
            logger.warning(
                "No tokens parsed from input, returning empty HeaderDocs")
        return HeaderDocs.from_tokens(tokens)

    def as_texts(self) -> List[str]:
        texts: List[str] = []

        def traverse(node: Union[HeaderNode, TextNode]) -> None:
            logger.debug(
                f"Traversing node {node.id}, type={node.type}, header={node.header}")
            if isinstance(node, HeaderNode):
                texts.append(node.get_text() if node.get_text() else "")
                for child in node.children:
                    traverse(child)
            else:
                texts.append(node.get_text() if node.get_text() else "")
        for node in self.root:
            traverse(node)
        logger.debug(f"Collected {len(texts)} texts")
        return texts

    def as_nodes(self) -> Nodes:
        nodes: Nodes = []

        def traverse(node: Union[HeaderNode, TextNode]) -> None:
            if isinstance(node, HeaderNode):
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

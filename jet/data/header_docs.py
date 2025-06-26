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

from jet.code.markdown_utils import parse_markdown

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
    title: str
    level: int
    children: List[Union['HeaderNode', TextNode]] = Field(default_factory=list)


Nodes = List[Union[HeaderNode, TextNode]]

# Header Tree to manage the hierarchy


class HeaderDocs(BaseModel):
    root: Nodes = Field(default_factory=list)

    @staticmethod
    def derive_text(token: MarkdownToken) -> str:
        """
        Derives the Markdown text representation for a given token based on its type.
        """
        if token['type'] == 'header' and token['level'] is not None:
            return f"{'#' * token['level']} {token['content'].strip()}" if token['content'] else ""

        elif token['type'] in ['unordered_list', 'ordered_list']:
            if not token['meta'] or 'items' not in token['meta']:
                return ""
            items = token['meta']['items']
            prefix = '*' if token['type'] == 'unordered_list' else lambda i: f"{i+1}."
            lines = []
            for i, item in enumerate(items):
                checkbox = '[x]' if item.get('checked', False) else '[ ]' if item.get(
                    'task_item', False) else ''
                prefix_str = prefix(
                    i) if token['type'] == 'ordered_list' else prefix
                line = f"{prefix_str} {checkbox}{' ' if checkbox else ''}{item['text']}".strip(
                )
                lines.append(line)
            return '\n'.join(lines)

        elif token['type'] == 'table':
            if not token['meta'] or 'header' not in token['meta'] or 'rows' not in token['meta']:
                return ""
            header = token['meta']['header']
            rows = token['meta']['rows']
            widths = [max(len(str(cell)) for row in [header] +
                          rows for cell in row[i:i+1]) for i in range(len(header))]
            lines = ['| ' + ' | '.join(cell.ljust(widths[i])
                                       for i, cell in enumerate(header)) + ' |']
            lines.append(
                '| ' + ' | '.join('-' * width for width in widths) + ' |')
            for row in rows:
                lines.append(
                    '| ' + ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + ' |')
            return '\n'.join(lines)

        elif token['type'] == 'code':
            language = token['meta'].get(
                'language', '') if token['meta'] else ''
            return f"```{' ' + language if language else ''}\n{token['content']}\n```"

        else:  # paragraph, blockquote, html_block
            return token['content']

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
                    title=token['content'],
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
                    content=HeaderDocs.derive_text(token),
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
                    f"{'#' * node.level} {node.title.strip()}" if node.title else "")
                for child in node.children:
                    traverse(child)
            else:
                texts.append(node.content)

        for node in self.root:
            traverse(node)
        return texts

    def as_nodes(self) -> Nodes:
        """
        Returns a flattened list of all nodes in document order.
        """
        nodes: Nodes = []

        def traverse(node: Union[HeaderNode, TextNode]) -> None:
            nodes.append(node)
            if isinstance(node, HeaderNode):
                for child in node.children:
                    traverse(child)

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
                    "title": node.title,
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

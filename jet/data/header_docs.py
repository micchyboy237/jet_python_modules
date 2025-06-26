from pathlib import Path
from typing import List, Optional, Union, Literal, Dict, Any, Tuple
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
import numpy as np
from numpy.linalg import norm

from jet.code.markdown_utils import derive_text, parse_markdown


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


class EmbeddingModel:
    """Simple embedding model for text (placeholder for a real model like BERT)."""

    @staticmethod
    def generate_embedding(text: str) -> np.ndarray:
        """
        Generates a simple word-based embedding for the input text.
        This is a placeholder implementation using character sum for demonstration.
        """
        # Simple embedding: sum of ASCII values normalized, replace with real model in production
        if not text:
            return np.zeros(128)
        vector = np.zeros(128)
        for char in text.lower()[:1000]:  # Limit text length for consistency
            idx = ord(char) % 128
            vector[idx] += 1
        return vector / (norm(vector) + 1e-10)  # Normalize with small epsilon


class VectorIndex(BaseModel):
    """Manages vector embeddings for nodes and performs vector search."""

    embeddings: Dict[str, np.ndarray] = Field(default_factory=dict)
    model: EmbeddingModel = Field(default_factory=EmbeddingModel)

    def add_node(self, node: Union[HeaderNode, TextNode]) -> None:
        """Generate and store embedding for a node."""
        text = node.title if isinstance(node, HeaderNode) else node.content
        self.embeddings[node.id] = self.model.generate_embedding(text)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for top_k most similar nodes to the query based on cosine similarity."""
        query_embedding = self.model.generate_embedding(query)
        scores = []

        for node_id, embedding in self.embeddings.items():
            # Cosine similarity: dot product of normalized vectors
            similarity = np.dot(query_embedding, embedding) / (
                (norm(query_embedding) * norm(embedding)) + 1e-10
            )
            scores.append((node_id, float(similarity)))

        # Sort by similarity (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HeaderDocs(BaseModel):
    root: Nodes = Field(default_factory=list)
    vector_index: VectorIndex = Field(default_factory=VectorIndex)

    @staticmethod
    def from_tokens(tokens: List[MarkdownToken]) -> 'HeaderDocs':
        """
        Converts a list of MarkdownToken into a HeaderDocs with parent-child relationships
        and builds vector index for search.
        """
        root: Nodes = []
        parent_stack: List[HeaderNode] = []
        seen_ids: set = set()
        vector_index = VectorIndex()

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
                vector_index.add_node(new_header)  # Add to vector index

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
                vector_index.add_node(text_node)  # Add to vector index

                if parent_stack:
                    parent_stack[-1].children.append(text_node)
                else:
                    root.append(text_node)

        return HeaderDocs(root=root, vector_index=vector_index)

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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Union[HeaderNode, TextNode], float]]:
        """Search for nodes matching the query using vector similarity."""
        results = self.vector_index.search(query, top_k)
        nodes = {node.id: node for node in self.as_nodes()}  # Map IDs to nodes
        return [(nodes[node_id], score) for node_id, score in results if node_id in nodes]

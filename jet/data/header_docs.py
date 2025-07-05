from pathlib import Path
from typing import List, Optional, TypedDict, Union, Literal, Dict, Any
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from jet.code.markdown_types import (
    MarkdownToken,
)
from jet.code.markdown_utils import derive_text, parse_markdown
import uuid
from jet.data.header_types import Nodes, TextNode, HeaderNode
from jet.logger import logger
from jet.models.model_types import ModelType
from jet.models.tokenizer.base import get_tokenizer, count_tokens


class HeaderDict(TypedDict):
    parent_header: Optional[str]
    header: str
    content: str
    level: int


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
        for doc_index, token in enumerate(tokens):
            logger.debug(
                f"Processing token type: {token['type']}, content: {token['content'][:50]}...")
            if token['type'] == 'header' and token['level'] is not None:
                header_id = generate_unique_id()
                text = derive_text(token)
                header_lines = text.splitlines()
                # Preserve hashtags in header by using the original content with level
                header = f"{'#' * token['level']} {header_lines[0].lstrip('#').strip()}"
                content = "\n".join(header_lines[1:]).strip()
                logger.debug(
                    f"Creating HeaderNode: header={header}, content={content}, level={token['level']}")
                new_header = HeaderNode(
                    doc_index=doc_index,
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
                    doc_index=doc_index,
                    type=token['type'],
                    header=header,
                    content=content,
                    meta=meta,
                    line=token['line'],
                    parent_id=parent_stack[-1].id if parent_stack else None,
                    id=text_id,
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
    def from_texts(texts: List[str]) -> 'HeaderDocs':
        """
        Create a HeaderDocs instance from a list of text strings, each parsed as markdown.

        Args:
            texts: List of strings to parse as markdown.

        Returns:
            HeaderDocs instance with parsed tokens and node structure.
        """
        logger.debug(f"Processing {len(texts)} text inputs")
        all_tokens: List[MarkdownToken] = []

        for idx, text in enumerate(texts):
            logger.debug(
                f"Parsing text {idx + 1}/{len(texts)}, content: {text[:100]}...")
            tokens = parse_markdown(text)
            if not tokens:
                logger.warning(f"No tokens parsed from text {idx + 1}")
                continue
            all_tokens.extend(tokens)
            logger.debug(f"Parsed {len(tokens)} tokens from text {idx + 1}")

        if not all_tokens:
            logger.warning(
                "No valid tokens parsed from any input texts, returning empty HeaderDocs")
            return HeaderDocs(root=[], tokens=[])

        logger.debug(f"Total {len(all_tokens)} tokens parsed from all texts")
        return HeaderDocs.from_tokens(all_tokens)

    @staticmethod
    def from_dicts(dict_list: List[HeaderDict]) -> 'HeaderDocs':
        logger.debug(f"Processing {len(dict_list)} dictionary inputs")
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

        # Track headers to find parents
        header_to_id: Dict[str, str] = {}

        for idx, item in enumerate(dict_list):
            logger.debug(
                f"Processing dictionary {idx + 1}/{len(dict_list)}: header={item['header']}")

            # Create HeaderNode
            header_id = generate_unique_id()
            header_node = HeaderNode(
                doc_index=idx,
                parent_header=item['parent_header'],
                header=item['header'],
                content=item["content"],
                level=item["level"],
                line=idx * 2,
                id=header_id,
            )
            id_to_node[header_id] = header_node
            header_to_id[item['header']] = header_id

            # Handle parent relationship
            if item['parent_header']:
                parent_id = header_to_id.get(item['parent_header'])
                if parent_id and parent_id in id_to_node:
                    header_node.parent_id = parent_id
                    header_node._parent_node = id_to_node[parent_id]
                    header_node.parent_header = item['parent_header']
                    id_to_node[parent_id].children.append(header_node)
                    logger.debug(
                        f"Linked header {header_id} to parent {parent_id}")

                    # Adjust parent stack to maintain correct hierarchy
                    while parent_stack and parent_stack[-1].level >= header_node.level:
                        parent_stack.pop()
                    if parent_stack and parent_stack[-1].id != parent_id:
                        parent_stack.append(id_to_node[parent_id])
                else:
                    logger.warning(
                        f"Parent header '{item['parent_header']}' not found for header '{item['header']}'")
                    root.append(header_node)
            else:
                root.append(header_node)

            parent_stack.append(header_node)
            logger.debug(
                f"Added header {header_id} to {'root' if not item['parent_header'] else f'parent {parent_id}'}")

            # # Create TextNode
            # if item['content']:
            #     text_id = generate_unique_id()
            #     text_node = TextNode(
            #         doc_index=idx,
            #         type="paragraph",
            #         header=item['header'],
            #         parent_header=item['parent_header'],
            #         parent_id=header_id,
            #         content=item['content'],
            #         meta={},
            #         line=idx * 2 + 1,
            #         id=text_id,
            #         level=header_node.level,  # Set TextNode level to match parent HeaderNode
            #     )
            #     id_to_node[text_id] = text_node
            #     text_node._parent_node = header_node
            #     text_node.parent_header = header_node.header
            #     header_node.children.append(text_node)
            #     logger.debug(
            #         f"Added text node {text_id} as child of {header_id} with level {text_node.level}")

        # Create tokens for compatibility
        tokens: List[MarkdownToken] = []
        for idx, item in enumerate(dict_list):
            tokens.append({
                "type": "header",
                "content": item['header'],
                "level": id_to_node[header_to_id[item['header']]].level,
                "meta": {},
                "line": idx * 2
            })
            # if item['text']:
            #     tokens.append({
            #         "type": "paragraph",
            #         "content": item['text'],
            #         "level": None,
            #         "meta": {},
            #         "line": idx * 2 + 1
            #     })

        logger.debug(
            f"Created HeaderDocs with {len(root)} root nodes and {len(tokens)} tokens")
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
            # Ensure doc_id is set
            return HeaderDocs(root=[], tokens=[])
        return HeaderDocs.from_tokens(tokens)

    def calculate_num_tokens(self, model_name_or_tokenizer: Union[ModelType, Tokenizer]) -> List[int]:
        nodes = self.as_nodes()
        texts = [node.get_text() for node in nodes]
        logger.debug(f"Calculating token counts for texts: {texts}")
        token_counts = count_tokens(
            model_name_or_tokenizer, texts, prevent_total=True)
        logger.debug(f"Token counts returned: {token_counts}")
        if isinstance(token_counts, int):
            logger.debug("No tokens to process, token_counts is an integer")
            return []

        def update_node_tokens(node: Union[HeaderNode, TextNode], index: List[int]) -> None:
            if index[0] >= len(token_counts):
                logger.error(
                    f"Index out of range: {index[0]} >= {len(token_counts)} for node {node.id}")
                return
            node.num_tokens = token_counts[index[0]]
            logger.debug(
                f"Updated node {node.id} with {node.num_tokens} tokens")
            index[0] += 1
            if isinstance(node, HeaderNode):
                for child in node.children:
                    update_node_tokens(child, index)
        index = [0]
        for node in self.root:  # Use self.root directly to update original nodes
            update_node_tokens(node, index)
        logger.debug(
            f"Finished updating token counts, processed {index[0]} nodes")
        return token_counts

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
            nodes.append(node)  # Reference original node
            if isinstance(node, HeaderNode):
                for child in node.children:
                    traverse(child)
        for node in self.root:
            traverse(node)
        return nodes

    def as_tree(self) -> Dict[str, Any]:
        def node_to_dict(node: Union[HeaderNode, TextNode]) -> Dict[str, Any]:
            base = {
                "id": node.id,
                "doc_id": node.doc_id,  # Include doc_id
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

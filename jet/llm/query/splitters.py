from typing import Optional
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode, IndexNode


def split_sub_nodes(
    base_nodes: list[BaseNode],
    chunk_sizes: list[int] = [128, 256, 512],
    chunk_overlap: int = 20,
) -> list[BaseNode]:
    sub_node_parsers = [
        SentenceSplitter(chunk_size=c, chunk_overlap=chunk_overlap) for c in chunk_sizes
    ]
    all_nodes: list[BaseNode] = []

    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    return all_nodes


def split_heirarchical_nodes(
    base_nodes: list[BaseNode],
    chunk_sizes: list[int] = [512, 256, 128],
    chunk_overlap: int = 20,
) -> list[BaseNode]:
    chunk_sizes = sorted(chunk_sizes, reverse=True)

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes, chunk_overlap=chunk_overlap)
    all_nodes = node_parser.get_nodes_from_documents(
        base_nodes, show_progress=True)

    return all_nodes

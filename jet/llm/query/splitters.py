from typing import Optional
from jet.code.splitter_markdown_utils import HeaderNode, get_flat_header_list, get_header_contents
from jet.file.utils import load_file
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document, BaseNode, IndexNode, TextNode


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


def split_markdown_header_nodes(
    base_nodes: list[BaseNode] | list[Document],
    include_child_contents: bool = True
) -> list[BaseNode]:
    all_nodes: list[BaseNode] = []
    contents = [
        {
            "metadata": node.metadata,
            "content": node.text,
        }
        for node in base_nodes
    ]

    for item in contents:
        file_metadata = item["metadata"]
        md_text = item["content"]

        header_contents = get_header_contents(
            md_text, include_child_contents=include_child_contents)
        all_header_nodes = get_flat_header_list(header_contents)
        all_header_nodes: list[HeaderNode] = [
            {**item, "metadata": {**file_metadata, **item["metadata"]}} for item in all_header_nodes]
        # filtered_header_contents = [
        #     item for item in header_contents if item['details'].strip()]

        nodes = [TextNode(text=item["content"], metadata=item["metadata"])
                 for item in all_header_nodes]
        all_nodes.extend(nodes)

    return all_nodes


def update_header_contents_metadata(header_nodes: list[HeaderNode]) -> list[dict]:

    md_text = "\n".join([item["content"] for item in header_nodes])

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False)
    md_header_splits = markdown_splitter.split_text(md_text)
    md_header_contents = []
    for split_idx, split in enumerate(md_header_splits):
        content = split.page_content
        # metadata = split.metadata

        # Remove placeholder text
        content = content.replace("<placeholder>", "")

        # Remove unwanted trailing line spaces
        content_lines = [line.rstrip() for line in content.splitlines()]
        content = "\n".join(content_lines)

        content = content.strip()

        header_group = header_groups[split_idx]

        md_header_contents.append({
            "heading": header_group["header"],
            "details": header_group["details"],
            "content": content,
            "length": len(content),
            "metadata": {
                **header_group["metadata"],
                "tags": list(split.metadata.values())
            }
        })
    return md_header_contents


def get_header_level(header: str) -> int:
    """Get the header level of a markdown header or HTML header tag."""
    if header.startswith("#"):
        header_level = 0
        for c in header:
            if c == "#":
                header_level += 1
            else:
                break
        return header_level
    elif header.startswith("h") and header[1].isdigit() and 1 <= int(header[1]) <= 6:
        return int(header[1])
    else:
        raise ValueError(f"Invalid header format: {header}")

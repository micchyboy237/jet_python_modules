from typing import Optional
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
    chunk_overlap: int = 100,
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

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]

        header_contents = get_header_contents(md_text, headers_to_split_on)
        header_contents = [{**item, "metadata": {**item["metadata"],
                                                 **file_metadata}} for item in header_contents]
        # filtered_header_contents = [
        #     item for item in header_contents if item['details'].strip()]

        nodes = [TextNode(text=item["content"], metadata=item["metadata"])
                 for item in header_contents]
        all_nodes.extend(nodes)

    return all_nodes


def get_header_contents(md_text: str, headers_to_split_on: list[tuple[str, str]] = []) -> list[dict]:
    header_lines = []
    header_prefixes = [f"{prefix.strip()} " for prefix,
                       _ in headers_to_split_on]
    all_lines = md_text.splitlines()
    for line_idx, line in enumerate(all_lines):
        if any(line.lstrip().startswith(prefix) for prefix in header_prefixes):
            header_lines.append({"index": line_idx, "line": line})

    header_content_indexes = [item["index"]
                              for item in header_lines] + [len(all_lines)]
    header_content_ranges = [(header_content_indexes[item_idx], header_content_indexes[item_idx + 1])
                             for item_idx, _ in enumerate(header_lines)]
    header_groups = []
    previous_added_lines = 0
    for start_idx, end_idx in header_content_ranges:
        start_idx += previous_added_lines
        end_idx += previous_added_lines
        header_line, *contents = all_lines[start_idx: end_idx]
        header_level = get_header_level(header_line)
        content = "\n".join(contents)

        start_line_idx = start_idx - previous_added_lines
        end_line_idx = end_idx - previous_added_lines

        # if not content.strip():
        #     lines_to_insert = ["", "<placeholder>", ""]
        #     previous_added_lines += len(lines_to_insert)
        #     all_lines[end_idx:end_idx] = lines_to_insert  # Inserts these lines

        details = content if content else "<placeholder>"
        block_content = f"{header_line}\n\n{details}\n\n"

        header_groups.append({
            "header": header_line,
            "details": content,
            "content": block_content,
            "metadata": {
                "start_line_idx": start_line_idx,
                "end_line_idx": end_line_idx,
                "depth": header_level,
            }
        })

    md_text = "\n".join([item["content"] for item in header_groups])

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

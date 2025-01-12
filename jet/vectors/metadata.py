from typing import AsyncGenerator
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import TextNode


# Generate metadata references
async def generate_metadata(
    base_nodes: list[TextNode],
    extractors: list[BaseExtractor],
) -> AsyncGenerator[tuple[TextNode, list], None]:
    for node in base_nodes:
        extracted_results = []
        for extractor in extractors:
            # Passing the node as a list
            metadata = await extractor.aextract([node])
            # Assuming aextract returns a list
            extracted_results.append(metadata)
        yield node, extracted_results


# Parse nodes from documents
def parse_nodes(docs, chunk_size=1024, chunk_overlap=200):
    parser = SentenceSplitter(chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    for idx, node in enumerate(nodes):
        node.id_ = f"node-{idx}"
    return nodes

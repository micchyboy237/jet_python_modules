from llama_index.core.schema import BaseNode, TextNode, ImageNode, NodeWithScore
from llama_index.core.base.response.schema import Response
from jet.vectors.node_types import SourceNodeAttributes


def get_source_node_attributes(
    source_node: NodeWithScore | BaseNode | Response,
) -> SourceNodeAttributes:
    attributes: SourceNodeAttributes = {}

    score = None

    if hasattr(source_node, "score"):
        score = source_node.score
        attributes["score"] = score

    if isinstance(source_node.node, BaseNode):
        source_node = source_node.node

    attributes["node_id"] = source_node.node_id
    attributes["metadata"] = source_node.metadata

    if isinstance(source_node, TextNode):
        attributes["text"] = source_node.text
        attributes["text_length"] = len(source_node.text)

        if hasattr(source_node, "start_char_idx") and hasattr(source_node, "end_char_idx"):
            attributes["start_end"] = (
                source_node.start_char_idx or 0, source_node.end_char_idx or attributes["text_length"] - 1)

    if isinstance(source_node, ImageNode):
        attributes["image_info"] = {
            "image": source_node.image,
            "image_path": source_node.image_path,
            "image_url": source_node.image_url,
            "image_mimetype": source_node.image_mimetype,
        }

    return attributes

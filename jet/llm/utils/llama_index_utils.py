import json
from llama_index.core.schema import BaseNode, TextNode, ImageNode, NodeWithScore
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from jet.logger import logger


def display_jet_source_node(
    source_node: NodeWithScore | BaseNode | Response,
    source_length: int = 50,
    show_source_metadata: bool = True,
) -> None:
    score = None

    if hasattr(source_node, "score"):
        score = source_node.score

    if isinstance(source_node.node, BaseNode):
        source_node = source_node.node

    logger.log("Node ID:", source_node.node_id, colors=["GRAY", "INFO"])

    # if show_source_metadata:
    #     logger.log("Metadata:", source_node.metadata, colors=[
    #                "GRAY", "INFO"])
    if 'file_name' in source_node.metadata:
        logger.log("File name:", source_node.metadata['file_name'],
                   colors=["GRAY", "SUCCESS"])
    if 'chunk_size' in source_node.metadata:
        logger.log("Chunk size:", source_node.metadata['chunk_size'],
                   colors=["GRAY", "SUCCESS"])
    if 'depth' in source_node.metadata:
        logger.log("Depth:", source_node.metadata['depth'],
                   colors=["GRAY", "SUCCESS"])

    if isinstance(source_node, TextNode):
        logger.log(
            "Text:",
            (source_node.text[:source_length] + '...') if source_length and len(
                source_node.text) > source_length else source_node.text,
            colors=["GRAY", "DEBUG"],
        )

        if 'file_path' in source_node.metadata:
            with open(source_node.metadata['file_path'], "r", encoding="utf-8") as file:
                content_length = len(file.read())
            logger.log("File length:", content_length,
                       colors=["GRAY", "SUCCESS"])
        logger.log("Text length:", len(source_node.text),
                   colors=["GRAY", "SUCCESS"])
        if hasattr(source_node, "start_char_idx") and hasattr(source_node, "end_char_idx"):
            logger.log("Start - End:", source_node.start_char_idx, "-", source_node.end_char_idx,
                       colors=["GRAY", "SUCCESS"])

    if isinstance(source_node, ImageNode):
        logger.log("Image:")
        logger.debug(json.dumps({
            "image": source_node.image,
            "image_path": source_node.image_path,
            "image_url": source_node.image_url,
            "image_mimetype": source_node.image_mimetype,
        }, indent=2))

    if score != None:
        logger.log("Similarity:", f"{score:.4f}", colors=["GRAY", "SUCCESS"])


def display_jet_source_nodes(
    query: str,
    source_nodes: list[NodeWithScore | BaseNode] | RESPONSE_TYPE,
    source_length: int = 50,
    show_source_metadata: bool = True,
):
    logger.newline()
    logger.log("Query:", query, colors=["WHITE", "INFO"])

    if isinstance(source_nodes, Response):
        response = source_nodes
        logger.log("Response:", response.response, colors=["WHITE", "SUCCESS"])
        # if response.metadata:
        #     logger.log("Response Metadata:", response.metadata,
        #                colors=["WHITE", "DEBUG"])
        source_nodes = response.source_nodes

    logger.log("Nodes Count:", len(source_nodes), colors=["WHITE", "SUCCESS"])
    # for idx, source_node in enumerate(source_nodes):
    #     logger.newline()
    #     logger.info(f"Node {idx + 1}:")
    #     # logger.debug(json.dumps(make_serializable(source_node), indent=2))
    #     display_jet_source_node(
    #         source_node,
    #         source_length,
    #         show_source_metadata,
    #     )
    #     logger.log("---")
    # logger.newline()

    if isinstance(source_nodes[0], NodeWithScore):
        for idx, source_node in enumerate(source_nodes):
            score = source_node.score if source_node.score != None else 0
            content = json.dumps(source_node.text).strip('"')
            logger.log(
                f"{idx + 1}:",
                source_node.metadata['file_name'] if 'file_name' in source_node.metadata else str(
                    source_node.metadata),
                f"{score:.4f}",
                colors=["INFO", "DEBUG", "SUCCESS"]
            )
            logger.log(content[:source_length] + '...')


# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample text node content that demonstrates how text can be expanded to meet specific word count requirements. By elaborating on the original content, we can increase the length while maintaining focus. This approach is useful for fulfilling word count tasks or creating longer content pieces, effectively extending text without losing its purpose."
    # Create a sample TextNode (replace with real data)
    text_node = TextNode(
        node_id="1",
        text=sample_text,
        metadata={"key": "value"},
    )

    # Create a sample NodeWithScore object
    text_source_node = NodeWithScore(
        node=text_node,  # or any other node type
        score=0.80
    )

    # Create a sample ImageNode (replace with real data)
    image_node = ImageNode(
        node_id="2",
        metadata={"key": "value"},
        image="path/to/sample_image.jpg"
    )

    # Create a sample NodeWithScore object
    image_source_node = NodeWithScore(
        node=image_node,  # or any other node type
        score=0.95
    )

    # Call the display_jet_source_node function with sample data
    # display_jet_source_node(source_node=text_source_node)

    # Call the display_jet_source_node function with sample data
    # display_jet_source_node(source_node=image_source_node)

    # Call the display_jet_source_nodes function with sample data
    display_jet_source_nodes([text_source_node, image_source_node])

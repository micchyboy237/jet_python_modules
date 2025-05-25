import logging
from llama_index.core.schema import NodeRelationship
from jet.vectors.document_types import HeaderTextNode
from typing import Optional, Callable, Union
from typing import Callable, Literal, Optional, TypedDict, Union
from jet.llm.embeddings.sentence_embedding import get_tokenizer_fn
from jet.llm.mlx.models import get_embedding_size
from jet.logger import logger
from jet.utils.doc_utils import add_parent_child_relationship, add_sibling_relationship
from jet.vectors.document_types import HeaderDocument, HeaderTextNode
from jet.wordnet.words import get_words
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
import tiktoken
from jet.llm.llm_types import Message
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from jet.llm.models import (
    OLLAMA_EMBED_MODELS,
    OLLAMA_HF_MODEL_NAMES,
    OLLAMA_HF_MODELS,
    OLLAMA_LLM_MODELS,
    OLLAMA_MODEL_EMBEDDING_TOKENS,
    OLLAMA_MODEL_NAMES,
)
from jet.llm.mlx.mlx_types import ModelType


def get_ollama_models():
    """Lazy loading of Ollama models to avoid circular imports"""

    return OLLAMA_HF_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS


def get_ollama_tokenizer(model_name: str | OLLAMA_MODEL_NAMES | OLLAMA_HF_MODEL_NAMES) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    if model_name in OLLAMA_MODEL_NAMES.__args__:
        model_name = OLLAMA_HF_MODELS[model_name]

    if model_name in OLLAMA_HF_MODEL_NAMES.__args__:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name)
        return tokenizer

    raise ValueError(f"Model \"{model_name}\" not found")


def get_tokenizer(model_name: str | OLLAMA_MODEL_NAMES | OLLAMA_HF_MODEL_NAMES) -> PreTrainedTokenizer | PreTrainedTokenizerFast | tiktoken.Encoding:
    if model_name in OLLAMA_MODEL_NAMES.__args__:
        model_name = OLLAMA_HF_MODELS[model_name]

    if model_name in OLLAMA_HF_MODEL_NAMES.__args__:
        return get_ollama_tokenizer(model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding


def tokenize(model_name: str | OLLAMA_MODEL_NAMES, text: str | list[str] | list[dict]) -> list[int] | list[list[int]]:
    tokenizer = get_tokenizer(model_name)

    if isinstance(text, list):
        texts = [str(t) for t in text]

        if isinstance(tokenizer, tiktoken.Encoding):
            tokenized = tokenizer.encode_batch(texts)
        else:
            tokenized = tokenizer.batch_encode_plus(texts, return_tensors=None)
            tokenized = tokenized["input_ids"]
        return tokenized
    else:
        tokens = tokenizer.encode(str(text))
        return tokens


def token_counter(
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: Optional[str | OLLAMA_MODEL_NAMES] = "mistral",
    prevent_total: bool = False
) -> int | list[int]:
    if not text:
        return 0

    tokenized = tokenize(model, text)
    if isinstance(text, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


def get_model_by_max_predict(text: str, max_predict: int = 500, type: Literal["llm", "embed"] = "llm") -> OLLAMA_LLM_MODELS:
    """
    Returns the first OLLAMA model (sorted by max tokens) that can accommodate
    the given text plus max_predict tokens. Raises error if none fits.
    """
    models = OLLAMA_LLM_MODELS.__args__ if type == "llm" else OLLAMA_EMBED_MODELS.__args__

    sorted_models = sorted(
        models,
        key=lambda name: OLLAMA_MODEL_EMBEDDING_TOKENS[name]
    )

    text_token_count: int = token_counter(text)

    for model in sorted_models:
        max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
        if text_token_count + max_predict <= max_tokens:
            return model

    raise ValueError(
        f"No suitable model found. Required tokens: {text_token_count + max_predict}, "
        f"but highest model max is {OLLAMA_MODEL_EMBEDDING_TOKENS[sorted_models[-1]]}"
    )


class TokenCountsInfoResult(TypedDict):
    tokens: int
    text: str


class TokenCountsInfo(TypedDict):
    average: float
    max: int
    min: int
    results: list[TokenCountsInfoResult]


def get_token_counts_info(texts: list[str], model: OLLAMA_MODEL_NAMES) -> TokenCountsInfo:
    token_counts: list[int] = token_counter(texts, model, prevent_total=True)
    total_count = sum(token_counts)
    avg_count = round(total_count / len(token_counts),
                      2) if token_counts else 0.0  # Rounded average
    results: list[TokenCountsInfoResult] = [{"tokens": count, "text": text}
                                            for count, text in zip(token_counts, texts)]

    return {
        "min": min(token_counts) if token_counts else 0,
        "max": max(token_counts) if token_counts else 0,
        "average": avg_count,
        "results": sorted(results, key=lambda x: x["tokens"])
    }


def get_model_max_tokens(
    model: Optional[str | OLLAMA_MODEL_NAMES] = "mistral",
) -> int:
    if model in OLLAMA_MODEL_EMBEDDING_TOKENS:
        return OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer.model_max_length
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.max_token_value


def filter_texts(
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: str | OLLAMA_MODEL_NAMES = "mistral",
    max_tokens: Optional[int | float] = None,
) -> str | list[str] | list[dict] | list[ChatMessage]:
    if not max_tokens:
        max_tokens = 0.5

    tokenizer = get_tokenizer(OLLAMA_HF_MODELS[model])
    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(
            get_model_max_tokens(model) * max_tokens)
    else:
        max_tokens = max_tokens or get_model_max_tokens(model)

    if isinstance(text, str):
        token_count = token_counter(text, model)
        if token_count <= max_tokens:
            return [text]

        # Split into manageable chunks
        tokens = tokenize(OLLAMA_HF_MODELS[model], text)
        return tokenizer.decode(tokens[0:max_tokens], skip_special_tokens=False)
    else:
        if isinstance(text[0], str):
            filtered_texts = []
            current_token_count = 0

            # Precompute token counts for all text in a single batch for efficiency
            text_token_counts = token_counter(text, model, prevent_total=True)

            for t, token_count in zip(text, text_token_counts):
                # Check if adding this text will exceed the max_tokens limit
                if current_token_count + token_count <= max_tokens:
                    filtered_texts.append(t)
                    current_token_count += token_count
                else:
                    break  # Stop early since texts are already sorted by score

            return filtered_texts
        else:
            messages = text.copy()
            token_count = token_counter(str(messages), model)

            if isinstance(token_count, int) and token_count <= max_tokens:
                return messages

            # Remove messages one by one from second to last up to second
            while len(messages) > 2 and isinstance(token_count, int) and token_count > max_tokens:
                messages.pop(-2)  # Remove second to last message
                token_count = token_counter(str(messages), model)

            return messages


def group_texts(
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: str | OLLAMA_MODEL_NAMES = "mistral",
    max_tokens: Optional[int | float] = None,
) -> list[list[str]]:
    if not max_tokens:
        max_tokens = 0.5

    tokenizer = get_tokenizer(OLLAMA_HF_MODELS[model])
    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(get_model_max_tokens(model) * max_tokens)
    else:
        max_tokens = max_tokens or get_model_max_tokens(model)

    if isinstance(text, str):
        tokens = tokenize(OLLAMA_HF_MODELS[model], text)
        grouped_texts = []

        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i + max_tokens]
            grouped_texts.append(tokenizer.decode(
                chunk, skip_special_tokens=False))

        return grouped_texts

    elif isinstance(text, list) and isinstance(text[0], str):
        grouped_texts = []
        current_group = []
        current_token_count = 0

        text_token_counts = token_counter(text, model, prevent_total=True)

        for t, token_count in zip(text, text_token_counts):
            if current_token_count + token_count > max_tokens:
                grouped_texts.append(current_group)
                current_group = []
                current_token_count = 0

            current_group.append(t)
            current_token_count += token_count

        if current_group:
            grouped_texts.append(current_group)

        return grouped_texts

    else:
        raise TypeError("Unsupported input type for group_texts")


def group_nodes(
    nodes: list[TextNode] | list[NodeWithScore],
    model: str | OLLAMA_MODEL_NAMES = "mistral",
    # New argument to enforce minimum token count per group
    min_tokens: Optional[int | float] = None,
    max_tokens: Optional[int | float] = None,
) -> list[list[TextNode] | list[NodeWithScore]]:
    if not max_tokens:
        max_tokens = 0.5
    if not min_tokens:
        min_tokens = 0.5

    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(get_model_max_tokens(model) * max_tokens)
    else:
        max_tokens = max_tokens or get_model_max_tokens(model)

    if isinstance(min_tokens, float):
        min_tokens = int(max_tokens * min_tokens)

    grouped_nodes = []
    current_group = []
    current_token_count = 0

    text_token_counts = token_counter(
        [n.text for n in nodes], model, prevent_total=True)

    for node, token_count in zip(nodes, text_token_counts):
        # If adding this node exceeds the max token count, start a new group
        if current_token_count + token_count > max_tokens:
            # Add the current group, ensuring it's not too small
            if current_token_count >= min_tokens:
                grouped_nodes.append(current_group)
            else:
                # If it's too small, merge with the next group
                if grouped_nodes:
                    grouped_nodes[-1].extend(current_group)
                else:
                    grouped_nodes.append(current_group)

            current_group = []
            current_token_count = 0

        current_group.append(node)
        current_token_count += token_count

    # Add the last group, ensuring it meets the min_tokens requirement
    if current_group:
        if current_token_count >= min_tokens:
            grouped_nodes.append(current_group)
        else:
            if grouped_nodes:
                grouped_nodes[-1].extend(current_group)
            else:
                grouped_nodes.append(current_group)

    return grouped_nodes


def calculate_num_predict_ctx(prompt: str | list[str] | list[ChatMessage] | list[Message], model: str = "llama3.1", *, system: str = "", max_prediction_ratio: float = 0.75):
    user_tokens: int = token_counter(prompt, model)
    system_tokens: int = token_counter(system, model)
    prompt_tokens = user_tokens + system_tokens
    num_predict = int(prompt_tokens * max_prediction_ratio)
    num_ctx = prompt_tokens + num_predict

    model_max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    if num_ctx > model_max_tokens:
        raise ValueError({
            "prompt_tokens": prompt_tokens,
            "num_predict": num_predict,
            "error": f"Context window size ({num_ctx}) exceeds model's maximum tokens ({model_max_tokens})",
        })

    return {
        "user_tokens": user_tokens,
        "system_tokens": system_tokens,
        "prompt_tokens": prompt_tokens,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }


def truncate_texts(texts: str | list[str], model: str, max_tokens: int) -> list[str]:
    """
    Truncates texts that exceed the max_tokens limit.

    Args:
        texts (str | list[str]): A list of texts to be truncated.
        model (str): The model name for tokenization.
        max_tokens (int): The maximum number of tokens allowed per text.

    Returns:
        list[str]: A list of truncated texts.
    """
    tokenizer = get_tokenizer(model)

    if isinstance(texts, str):
        texts = [texts]

    tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors=None)
    tokenized_texts = tokenized_texts["input_ids"]
    truncated_texts = []

    for text, tokens in zip(texts, tokenized_texts):
        if len(tokens) > max_tokens:
            truncated_text = tokenizer.decode(
                tokens[:max_tokens], skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        else:
            truncated_texts.append(text)

    return truncated_texts


def split_texts(
    texts: str | list[str],
    model: str | OLLAMA_MODEL_NAMES,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    buffer: int = 0
) -> list[str]:
    """
    Splits a list of texts into smaller chunks based on chunk_size, chunk_overlap, and buffer.

    Args:
        texts (str | list[str]): List of input texts to be split.
        model (str): Model name for tokenization.
        chunk_size (int): Maximum tokens allowed per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.
        buffer (int, optional): Extra space reserved to avoid exceeding chunk_size. Default is 0.

    Returns:
        list[str]: A list of split text chunks.
    """
    if not chunk_size:
        chunk_size = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})")

    effective_max_tokens = max(chunk_size - buffer, 1)  # Ensure positive value
    if effective_max_tokens <= chunk_overlap:
        raise ValueError(
            f"Effective max tokens ({effective_max_tokens}) must be greater than chunk overlap ({chunk_overlap})")

    tokenizer = get_tokenizer(model)
    split_chunks = []

    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        tokens = tokenizer.encode(text) if hasattr(
            tokenizer, "encode") else tokenizer(text)
        total_tokens = len(tokens)

        if total_tokens <= effective_max_tokens:
            split_chunks.append(text)
            continue

        start = 0
        while start < total_tokens:
            end = min(start + effective_max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            try:
                chunk_text = tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True)
            except:
                chunk_text = tokenizer.decode(chunk_tokens)

            chunk_text = chunk_text.strip()

            if chunk_text:  # Ensure non-empty chunks are added
                split_chunks.append(chunk_text)

            if end == total_tokens:
                break
            start = max(end - chunk_overlap, 0)  # Prevent negative index

    logger.debug(
        f"Split {len(texts)} texts into {len(split_chunks)} chunks (buffer={buffer}, overlap={chunk_overlap}).")
    return split_chunks


def get_subtext_indices(text: str, subtext: str) -> tuple[int, int] | None:
    start = text.find(subtext)
    if start == -1:
        return None
    end = start + len(subtext) - 1
    return start, end


def split_docs(
    docs: Document | list[Document],
    model: Optional[str | OLLAMA_MODEL_NAMES] = None,
    chunk_size: int = 128,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Callable[[Union[str, list[str]]],
                                 Union[list[str], list[list[str]]]]] = None,
    tokens: Optional[list[int] | list[list[int]]] = None,
    buffer: int = 0
) -> list[TextNode]:
    if not isinstance(docs, list):
        docs = [docs]
    if tokens and not isinstance(tokens[0], list):
        tokens = [tokens]

    if tokens is not None:
        if len(tokens) != len(docs):
            raise ValueError(
                f"Length of provided tokens ({len(tokens)}) does not match number of documents ({len(docs)})"
            )
        tokens_matrix = tokens
        token_counts = [len(t) for t in tokens_matrix]
    else:
        if tokenizer:
            tokenizer_fn = tokenizer
        elif model:
            def _tokenizer(input):
                return tokenize(model, input)
            tokenizer_fn = _tokenizer
        else:
            tokenizer_fn = get_words

        doc_texts = [doc.text for doc in docs]
        tokens_matrix: list[list] = tokenizer_fn(doc_texts)
        token_counts: list[int] = [len(t) for t in tokens_matrix]

    if not chunk_size:
        if model:
            chunk_size = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
        else:
            average_tokens = sum(token_counts) / \
                len(token_counts) if token_counts else 0
            chunk_size = average_tokens

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})"
        )

    effective_max_tokens = max(chunk_size - buffer, 1)
    if effective_max_tokens <= chunk_overlap:
        raise ValueError(
            f"Effective max tokens ({effective_max_tokens}) must be greater than chunk_overlap ({chunk_overlap})"
        )

    nodes: list[TextNode] = []

    for doc, token_count in zip(docs, token_counts):
        node = TextNode(
            text=doc.text,
            metadata={
                **doc.metadata,
                "start_idx": 0,
                "end_idx": len(doc.text),
                "chunk_index": None,
            })

        if token_count > effective_max_tokens:
            splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splitted_texts = splitter.split_text(doc.text)

            prev_sibling: Optional[TextNode] = None
            last_pos = 0  # Track last position to handle overlapping or repeated subtexts
            # Start chunk_idx at 0 for sub-chunks
            for chunk_idx, subtext in enumerate(splitted_texts, start=0):
                # Find the next occurrence of subtext after last_pos
                start_idx = doc.text.find(subtext, last_pos)
                if start_idx == -1:
                    # If subtext not found, use last_pos as fallback
                    start_idx = last_pos
                end_idx = start_idx + len(subtext)
                last_pos = start_idx  # Update last_pos for next iteration

                sub_node = TextNode(
                    text=subtext,
                    metadata={
                        **doc.metadata,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_index": chunk_idx,
                    })
                nodes.append(sub_node)
                add_parent_child_relationship(
                    parent_node=node,
                    child_node=sub_node,
                )

                if prev_sibling:
                    add_sibling_relationship(prev_sibling, sub_node)

                prev_sibling = sub_node
        else:
            nodes.append(node)

    return nodes


def split_headers(
    docs: HeaderDocument | list[HeaderDocument],
    model: Optional[str | ModelType] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Callable[[Union[str, list[str]]],
                                 Union[list[str], list[list[str]]]]] = None,
    tokens: Optional[list[int] | list[list[int]]] = None,
    buffer: int = 0
) -> list[HeaderTextNode]:
    if not isinstance(docs, list):
        docs = [docs]
    if tokens and not isinstance(tokens[0], list):
        tokens = [tokens]

    # Set up tokenizer
    if tokenizer:
        tokenizer_fn = tokenizer
    elif model:
        tokenizer_fn = get_tokenizer_fn(model)
    else:
        tokenizer_fn = get_words  # Default tokenizer

    # Process tokens or tokenize documents
    if tokens is not None:
        if len(tokens) != len(docs):
            raise ValueError(
                f"Length of provided tokens ({len(tokens)}) does not match number of documents ({len(docs)})"
            )
        tokens_matrix = tokens
        token_counts = [len(t) for t in tokens_matrix]
    else:
        doc_texts = [doc.text for doc in docs]
        tokens_matrix: list[list] = tokenizer_fn(doc_texts)
        token_counts: list[int] = [len(t) for t in tokens_matrix]

    # Set chunk_size
    if not chunk_size:
        if model:
            chunk_size = get_embedding_size(model)
        else:
            average_tokens = sum(token_counts) / \
                len(token_counts) if token_counts else 0
            chunk_size = average_tokens

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})"
        )

    effective_max_tokens = max(chunk_size - buffer, 1)
    if effective_max_tokens <= chunk_overlap:
        raise ValueError(
            f"Effective max tokens ({effective_max_tokens}) must be greater than chunk_overlap ({chunk_overlap})"
        )

    nodes: list[HeaderTextNode] = []

    for doc_idx, (doc, token_count) in enumerate(zip(docs, token_counts)):
        # Create base node with original text and metadata
        node = HeaderTextNode(
            text=doc.text,
            metadata={
                **doc.metadata,
                # Use current node list length as doc_index
                "doc_index": len(nodes),
                "content": doc.metadata["content"],
                "start_idx": 0,
                "end_idx": len(doc.text),
                "chunk_index": None,
                "token_count": token_count,  # Add token count for unsplit document
                "texts": None,  # Initialize texts field
            }
        )

        if token_count > effective_max_tokens:
            splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splitted_texts = splitter.split_text(doc.metadata["content"])

            # Update parent node with split texts
            node.metadata["texts"] = splitted_texts

            prev_sibling: Optional[HeaderTextNode] = None
            last_pos = 0
            for chunk_idx, subtext in enumerate(splitted_texts, start=0):
                # Find the next occurrence of subtext after last_pos
                start_idx = doc.metadata["content"].find(subtext, last_pos)
                if start_idx == -1:
                    start_idx = last_pos
                end_idx = start_idx + len(subtext)
                last_pos = start_idx

                # Calculate token count for the chunk
                chunk_tokens = tokenizer_fn(subtext)
                chunk_token_count = len(chunk_tokens)

                # Create sub-node with updated content metadata
                sub_node = HeaderTextNode(
                    text=subtext,
                    metadata={
                        **doc.metadata,
                        # Use current node list length as doc_index
                        "doc_index": len(nodes),
                        "content": subtext,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_index": chunk_idx,
                        "token_count": chunk_token_count,  # Add token count for chunk
                        "texts": None,  # Child nodes don't need texts field
                    }
                )
                nodes.append(sub_node)
                add_parent_child_relationship(
                    parent_node=node,
                    child_node=sub_node,
                )

                if prev_sibling:
                    add_sibling_relationship(prev_sibling, sub_node)

                prev_sibling = sub_node
        else:
            nodes.append(node)

    return nodes


def merge_headers(
    nodes: list[HeaderTextNode],
    model_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    tokenizer: Optional[Callable[[Union[str, list[str]]],
                                 Union[list[int], list[list[int]]]]] = None,
    buffer: int = 0
) -> list[HeaderTextNode]:
    if not nodes:
        return []

    # Validate chunk_size and chunk_overlap
    if not chunk_size:
        texts = [node.text for node in nodes]
        token_counts = [len(text.split()) for text in texts]
        chunk_size = int(sum(token_counts) / len(token_counts)
                         ) if token_counts else 128

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})")

    effective_max_tokens = max(chunk_size - buffer, 1)
    if effective_max_tokens <= chunk_overlap:
        raise ValueError(
            f"Effective max tokens ({effective_max_tokens}) must be greater than chunk_overlap ({chunk_overlap})")

    # Set up tokenizer
    if tokenizer:
        tokenizer_fn = tokenizer
    elif model_id:
        try:
            tokenizer_fn = get_tokenizer_fn(model_id)
        except ImportError:
            raise ImportError(
                "Please install sentence_transformers: pip install sentence_transformers")
    else:
        raise ValueError("Either model_id or tokenizer must be provided")

    def encode_wrapper(texts: Union[str, list[str]]) -> Union[list[int], list[list[int]]]:
        result = tokenizer_fn.encode(
            texts, truncation=True, max_length=chunk_size)
        return result if isinstance(texts, list) else [result]

    def decode_wrapper(tokens: list[int], **kwargs) -> str:
        return tokenizer_fn.decode(tokens)

    # Compute token counts
    texts = [node.text for node in nodes]
    tokens_matrix = encode_wrapper(texts)
    token_counts = [len(tokens) for tokens in tokens_matrix]

    merged_nodes: list[HeaderTextNode] = []
    current_text = ""
    current_token_count = 0
    current_metadata = {}
    start_idx = 0
    chunk_idx = 0
    used_nodes: set[str] = set()
    i = 0

    while i < len(nodes):
        node = nodes[i]
        node_id = node.id_
        token_count = token_counts[i]

        if node_id in used_nodes:
            i += 1
            continue

        if current_token_count + token_count <= effective_max_tokens:
            current_text += (node.text + " ") if current_text else node.text
            current_token_count += token_count
            used_nodes.add(node_id)
            for key, value in node.metadata.items():
                if key not in current_metadata or not current_metadata[key]:
                    current_metadata[key] = value
            i += 1
            continue

        if current_text:
            end_idx = start_idx + len(current_text.strip())
            merged_node = HeaderTextNode(
                text=current_text.strip(),
                metadata={
                    **current_metadata,
                    "content": current_text.strip(),  # Set content to merged text
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_index": chunk_idx,
                }
            )
            merged_nodes.append(merged_node)
            chunk_idx += 1
            current_metadata = {}
            start_idx = end_idx

        if chunk_overlap > 0 and current_text:
            overlap_tokens = encode_wrapper(current_text)[-1][-chunk_overlap:]
            overlap_text = decode_wrapper(
                overlap_tokens, skip_special_tokens=True).strip()
            current_text = overlap_text
            current_token_count = len(overlap_tokens)
        else:
            current_text = ""
            current_token_count = 0

        if token_count > effective_max_tokens:
            splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splitted_texts = splitter.split_text(node.text)
            for sub_idx, subtext in enumerate(splitted_texts):
                sub_node = HeaderTextNode(
                    text=subtext,
                    metadata={
                        **node.metadata,
                        "content": subtext,  # Set content to subtext
                        "start_idx": start_idx,
                        "end_idx": start_idx + len(subtext),
                        "chunk_index": chunk_idx,
                        "sub_chunk": sub_idx,
                    }
                )
                merged_nodes.append(sub_node)
                start_idx += len(subtext)
                chunk_idx += 1
            used_nodes.add(node_id)
            i += 1
            current_text = ""
            current_token_count = 0
            continue

        current_text = node.text
        current_token_count = token_count
        current_metadata = node.metadata.copy()
        current_metadata["content"] = node.text  # Ensure content matches text
        used_nodes.add(node_id)
        i += 1

    if current_text:
        end_idx = start_idx + len(current_text.strip())
        merged_node = HeaderTextNode(
            text=current_text.strip(),
            metadata={
                **current_metadata,
                "content": current_text.strip(),  # Set content to final text
                "start_idx": start_idx,
                "end_idx": end_idx,
                "chunk_index": chunk_idx,
            }
        )
        merged_nodes.append(merged_node)

    # Deduplicate based on text
    seen_texts: set[str] = set()
    unique_nodes: list[HeaderTextNode] = []
    for node in merged_nodes:
        if node.text not in seen_texts:
            seen_texts.add(node.text)
            unique_nodes.append(node)

    logging.debug(
        f"Merged {len(nodes)} HeaderTextNodes into {len(unique_nodes)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap}, buffer={buffer})."
    )
    return unique_nodes


if __name__ == "__main__":
    from jet.file.utils import load_file
    from jet.search.formatters import clean_string
    from jet.transformers.formatters import format_json

    # models = ["llama3.1"]
    models = ["paraphrase-MiniLM-L12-v2"]
    ollama_models = {}

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data = load_file(data_file)
    docs = []
    for item in data:
        cleaned_sentence = clean_string(item)
        docs.append(cleaned_sentence)

    sample_text = "Text 1, Text 2"
    sample_texts = docs

    logger.info("Count tokens for: str")
    for model_name in models:
        result = token_counter(sample_text, model_name)
        logger.log("Count:", format_json(result), colors=["DEBUG", "SUCCESS"])

    logger.info("Count tokens info for: str")
    for model_name in models:
        splitted_texts = split_texts(
            docs, model_name, chunk_size=200, chunk_overlap=50)
        result = get_token_counts_info(splitted_texts, model_name)
        logger.log("Count:", format_json(result), colors=["DEBUG", "SUCCESS"])

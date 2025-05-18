from jet.llm.mlx.utils import get_model_max_tokens
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Callable, List, Dict, Optional, TypedDict, Union
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.wordnet.sentence import split_sentences
from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import transformers  # Assuming tokenizer is from transformers


# def count_tokens(
#     tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
#     prompt: Union[str, List[int], mx.array, List[dict], List[str]],
#     system_prompt: Optional[str] = None,
#     prefill_response: Optional[str] = None,
#     ignore_chat_template: bool = False,
#     use_default_chat_template: bool = False,
#     chat_template_config: Optional[dict] = None,
# ) -> int:
#     """
#     Count the number of tokens in a prompt.

#     Args:
#         tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer to use.
#         prompt (Union[str, List[int], mx.array, List[dict], List[str]]): The input prompt as a string,
#             list of token IDs, mx.array, list of message dictionaries with role/content, or list of strings.
#         system_prompt (Optional[str]): Optional system prompt for chat template. Default: None.
#         prefill_response (Optional[str]): Optional prefill response for chat template. Default: None.
#         ignore_chat_template (bool): If True, bypass chat template and encode prompt directly. Default: False.
#         use_default_chat_template (bool): If True, use the tokenizer's default chat template. Default: False.
#         chat_template_config (Optional[dict]): Additional config for chat template as a dictionary. Default: None.

#     Returns:
#         int: The number of tokens in the prompt.
#     """
#     if not isinstance(tokenizer, TokenizerWrapper):
#         tokenizer = TokenizerWrapper(tokenizer)

#     # Handle tokenized input (list of integers or mx.array)
#     if isinstance(prompt, (list, mx.array)) and (isinstance(prompt, mx.array) or
#                                                  (isinstance(prompt, list) and all(isinstance(x, int) for x in prompt))):
#         if isinstance(prompt, list):
#             prompt = mx.array(prompt)
#         return prompt.size

#     # Set up chat template
#     if use_default_chat_template and tokenizer.chat_template is None:
#         tokenizer.chat_template = tokenizer.default_chat_template

#     template_kwargs = chat_template_config or {}

#     if not ignore_chat_template and tokenizer.chat_template is not None:
#         # Handle prompt as list of dictionaries (chat messages)
#         if isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
#             messages = [{"role": "system", "content": system_prompt}
#                         ] if system_prompt else []
#             messages.extend(prompt)
#         # Handle prompt as list of strings (each string is a user message)
#         elif isinstance(prompt, list) and all(isinstance(msg, str) for msg in prompt):
#             messages = [{"role": "system", "content": system_prompt}
#                         ] if system_prompt else []
#             messages.extend({"role": "user", "content": msg} for msg in prompt)
#         else:
#             # Handle prompt as single string
#             messages = [{"role": "system", "content": system_prompt}
#                         ] if system_prompt else []
#             messages.append({"role": "user", "content": prompt})

#         has_prefill = prefill_response is not None
#         if has_prefill:
#             messages.append({"role": "assistant", "content": prefill_response})

#         prompt_text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             continue_final_message=has_prefill,
#             add_generation_prompt=not has_prefill,
#             **template_kwargs,
#         )
#         # Encode without special tokens to match generate() behavior
#         tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
#     else:
#         # Encode directly with special tokens handling
#         if isinstance(prompt, list) and all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
#             # Concatenate content from messages if chat template is ignored
#             prompt = " ".join(msg["content"] for msg in prompt)
#         elif isinstance(prompt, list) and all(isinstance(msg, str) for msg in prompt):
#             # Concatenate strings if chat template is ignored
#             prompt = " ".join(prompt)
#         add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
#             tokenizer.bos_token)
#         tokens = tokenizer.encode(
#             prompt, add_special_tokens=add_special_tokens)

#     return len(tokens)

def count_tokens(model: LLMModelType, messages: str | List[str] | List[Dict], prevent_total: bool = False) -> int | list[int]:
    # return count_tokens(self.tokenizer, messages)
    if not messages:
        return 0

    if isinstance(messages, list):
        messages = [str(t) for t in messages]

    tokenize = get_tokenizer_fn(model)
    tokenized = tokenize(messages)
    if isinstance(messages, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


class Metadata(TypedDict):
    texts_count: int
    is_truncated: bool
    total_tokens: int
    min_tokens: int
    max_tokens: int
    ave_tokens: int


class MergeResult(TypedDict):
    texts: List[str]
    token_counts: List[int]
    tokens: List[List[int]]
    token_strings: List[List[str]]
    decoded_tokens: List[List[str]]
    metadata: Metadata


def merge_texts(
    text: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    skip_special_tokens: bool = True,
    max_length: Optional[int] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None
) -> MergeResult:
    # Encode the text into token IDs
    token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
    total_tokens: int = len(token_ids)

    # If max_length is None or greater than total tokens, no truncation needed
    if max_length is None or max_length >= total_tokens:
        token_strings: List[str] = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        # Use batch_decode to decode all token IDs at once
        decoded_tokens: List[str] = [
            dt for dt in tokenizer.batch_decode(
                [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
            ) if dt
        ]

        return {
            "texts": [text] if text else [],
            "token_counts": [len(token_ids)],
            "tokens": [token_ids],
            "token_strings": [token_strings],
            "decoded_tokens": [decoded_tokens],
            "metadata": {
                "texts_count": 1,
                "is_truncated": False,
                "total_tokens": total_tokens,
                "min_tokens": total_tokens,
                "max_tokens": total_tokens,
                "ave_tokens": total_tokens,
            }
        }

    # Get the decoded text to find sentence boundaries
    decoded_text: str = tokenizer.decode(
        token_ids, skip_special_tokens=skip_special_tokens
    )

    # Split text into sentences using NLTK
    sentences: List[str] = split_fn(
        decoded_text) if split_fn else split_sentences(decoded_text)

    # Initialize variables for grouping texts
    grouped_texts: List[str] = []
    grouped_token_ids: List[List[int]] = []
    selected_token_ids: List[int] = []
    current_token_count: int = 0
    current_group: List[str] = []

    for i, sentence in enumerate(sentences):
        sentence_token_ids: List[int] = tokenizer.encode(
            sentence, add_special_tokens=False
        )
        sentence_token_count: int = len(sentence_token_ids)

        # If sentence token count > max_length, just add it
        if not max_length or sentence_token_count > max_length:
            grouped_texts.append(sentence)
            grouped_token_ids.append(sentence_token_ids)
        # Check if adding the sentence exceeds max_length
        elif current_token_count + sentence_token_count <= max_length:
            selected_token_ids.extend(sentence_token_ids)
            current_token_count += sentence_token_count
            current_group.append(sentence)
        else:
            # If there's a current group, add it to grouped_texts and clear it
            if current_group:
                grouped_texts.append(" ".join(current_group))
                grouped_token_ids.append(selected_token_ids)
                current_group = []
                current_token_count = 0
                selected_token_ids = []

            # Try merging with the next sentence if possible
            remaining_tokens: int = max_length - current_token_count
            if remaining_tokens > 0 and i + 1 < len(sentences):
                next_sentence: str = sentences[i + 1]
                merged_sentence: str = sentence + " " + next_sentence
                merged_token_ids: List[int] = tokenizer.encode(
                    merged_sentence, add_special_tokens=False
                )

                if len(merged_token_ids) <= max_length - current_token_count:
                    selected_token_ids.extend(merged_token_ids)
                    current_token_count += len(merged_token_ids)
                    current_group.append(merged_sentence)
                    # Skip the next sentence since it's merged
                    sentences[i + 1] = ""
                    continue

            # If we can't merge or no space left, start a new group
            if remaining_tokens >= sentence_token_count:
                current_group = [sentence]
                selected_token_ids.extend(sentence_token_ids)
                current_token_count = sentence_token_count
            else:
                break

    # Add the final group if it exists
    if current_group:
        grouped_texts.append(" ".join(current_group))
        grouped_token_ids.append(selected_token_ids)

    grouped_decoded_tokens: List[List[str]] = []
    grouped_token_strings: List[List[str]] = []
    token_counts: List[int] = []
    for token_ids in grouped_token_ids:
        token_counts.append(len(token_ids))
        # Convert selected token IDs to token strings and decoded tokens
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        grouped_token_strings.append(token_strings)
        # Use batch_decode to decode all selected token IDs at once
        decoded_tokens = [
            dt for dt in tokenizer.batch_decode(
                [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
            ) if dt
        ]
        grouped_decoded_tokens.append(decoded_tokens)

    # Prepare metadata
    metadata: Metadata = {
        "texts_count": len(grouped_texts),
        "is_truncated": len(grouped_texts) > 1,
        "total_tokens": total_tokens,
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "ave_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
    }

    return {
        "texts": grouped_texts,
        "token_counts": token_counts,
        "tokens": grouped_token_ids,
        "token_strings": grouped_token_strings,
        "decoded_tokens": grouped_decoded_tokens,
        "metadata": metadata
    }


def tokenize_strings(text: Union[str, List[str]], model: LLMModelType) -> Union[str, list[str]]:
    tokenizer = get_tokenizer(model)
    if isinstance(text, str):
        token_ids = tokenizer.encode(
            text, add_special_tokens=False)
        return tokenizer.convert_ids_to_tokens(token_ids)
    else:
        token_ids_list = tokenizer.batch_encode_plus(
            text, add_special_tokens=False)["input_ids"]
        return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]


def get_tokenizer(model: LLMModelType) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    model_name = resolve_model(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizer_fn(model: LLMModelType) -> Callable[[Union[str, List[str]]], Union[List[str], List[List[str]]]]:
    tokenizer = get_tokenizer(model)

    def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            token_ids = tokenizer.encode(
                text, add_special_tokens=False)
            return tokenizer.convert_ids_to_tokens(token_ids)
        else:
            token_ids_list = tokenizer.batch_encode_plus(
                text, add_special_tokens=False)["input_ids"]
            return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]
    return _tokenizer


def chunk_text(text: Union[str, List[str]], n: Optional[int] = None, overlap: int = 0, model: Optional[LLMModelType] = None) -> List[str]:
    def chunk_tokens(tokens: List[str], tokenizer: Optional[PreTrainedTokenizer] = None) -> List[str]:
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if tokenizer:
                chunk_text = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(chunk_tokens),
                    skip_special_tokens=True
                )
            else:
                chunk_text = ''.join(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

    # Determine chunk size
    chunk_size = n
    if n is None:
        if model is None:
            chunk_size = 512  # Default chunk size when no model and n is not provided
        else:
            max_tokens = get_model_max_tokens(model)
            # Use 80% of max tokens as chunk size
            chunk_size = int(max_tokens * 0.8)

    if isinstance(text, str):
        if model is None:
            # Character-based chunking for single string
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        # Token-based chunking for single string
        tokenizer = get_tokenizer(model)
        tokens = get_tokenizer_fn(model)(text)
        return chunk_tokens(tokens, tokenizer)

    # Handle list of strings
    if model is None:
        # Character-based chunking for list
        chunks = []
        for t in text:
            chunks.extend([t[i:i + chunk_size]
                          for i in range(0, len(t), chunk_size - overlap)])
        return chunks

    # Batch tokenization for list
    tokenizer = get_tokenizer(model)
    tokenized_texts = get_tokenizer_fn(model)(text)
    chunks = []
    for tokens in tokenized_texts:
        chunks.extend(chunk_tokens(tokens, tokenizer))
    return chunks

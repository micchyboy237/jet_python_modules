import json
from typing import Any, Generator, List, Optional, TypedDict
from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from tqdm import tqdm

CHUNK_SIZE = 1024
OVERLAP = 128

SETTINGS = {
    "seed": 0,
    "temperature": 0,
    # "temperature": 0.6,
    # "top_k": 40,
    # "top_p": 0.85,
    "stop": None,
    "num_keep": 0,
    # "mirostat_tau": 5.0,
}

LOWER_CHAR_SUMMARY_MODEL = "mistral"
ROOT_MODEL = "llama3.2"
COMBINE_MODEL = "mistral"

LOWER_CHAR_SYSTEM_MESSAGE = (
    "You are an AI assistant specialized in summarizing structured content. "
    "Your goal is to generate fewer characters while retaining all relevant details. "
    "Format the output as structured markdown, maintaining proper headings and bullet points where applicable."
)

ROOT_SYSTEM_MESSAGE = (
    "You are an AI assistant specialized in summarizing unstructured content scraped from the internet. "
    "Your goal is to generate a clear, concise, and factual summary of the given information, "
    "ensuring that no false information is introduced and that all relevant details are retained. "
    "The summary should always be shorter than the original content, conveying the key points efficiently. "
    "Format the output as structured markdown, maintaining proper headings and bullet points where applicable."
)

COMBINE_SYSTEM_MESSAGE = (
    "You are an AI assistant refining and merging multiple summarized sections into a more concise and coherent summary. "
    "Your goal is to combine the given summaries while eliminating redundancy, ensuring factual accuracy, "
    "and retaining all relevant details. The final summary should always be shorter than the combined input. "
    "Ensure logical flow and clarity. Format the output in structured markdown where applicable."
)


class TokenCounts(TypedDict):
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


class LLMSettings(TypedDict):
    model: str
    num_predict: int
    num_ctx: int


class SummaryResult(TypedDict):
    system: str
    prompt: str
    response: str
    llm_settings: LLMSettings
    token_counts: TokenCounts


class SummaryResultInfo(TypedDict):
    depth: int
    summary: SummaryResult


class SummaryData(TypedDict):
    chunks: int
    last_depth: int
    final_summary: SummaryResultInfo
    other_summaries: list[SummaryResultInfo]


class SummaryTokens(TypedDict):
    summary: str
    tokens: int


def group_summaries(summaries: list[str], model: str, system: str = "", separator: str = "\n\n\n") -> list[SummaryTokens]:
    from jet.token.token_utils import token_counter

    max_prediction_ratio_full = 0.5

    model_max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
    max_tokens_per_group = int(
        model_max_tokens * (1 - max_prediction_ratio_full))

    formatted_summaries = [
        f"Summary {idx + 1}\n\n{text}"
        for idx, text in enumerate(summaries)
    ]

    token_counts: list[int] = token_counter(
        formatted_summaries, model, prevent_total=True)
    system_token_count: int = token_counter(system, model)
    separator_token_count: int = token_counter(separator, model)

    current_group_summaries = []
    current_group_tokens = [system_token_count]

    grouped_summaries: list[SummaryTokens] = []
    prev_group_summaries = []

    for idx, token_count in enumerate(token_counts):
        text = formatted_summaries[idx]
        if sum(current_group_tokens) + token_count < max_tokens_per_group:
            current_group_summaries.append(text)
            current_group_tokens.append(token_count)
        else:
            separator_multi = len(prev_group_summaries) - 1
            total_separator_token_count = separator_multi * separator_token_count
            grouped_summaries.append({
                "summary": separator.join(prev_group_summaries),
                "tokens": sum(current_group_tokens) + total_separator_token_count
            })

            current_group_summaries = [text]
            current_group_tokens = [system_token_count, token_count]

        prev_group_summaries = current_group_summaries.copy()

    if prev_group_summaries:
        separator_multi = len(prev_group_summaries) - 1
        total_separator_token_count = separator_multi * separator_token_count
        grouped_summaries.append({
            "summary": separator.join(prev_group_summaries),
            "tokens": sum(current_group_tokens) + total_separator_token_count
        })

    return grouped_summaries


def generate_summary(prompt: str, model: str = "llama3.1", system: str = "", options: OllamaChatOptions = {}) -> SummaryResult:
    from jet.token.token_utils import calculate_num_predict_ctx

    prompt = prompt.strip()
    system = system.strip()

    calc_result = calculate_num_predict_ctx(prompt, model, system=system)

    options = {
        **SETTINGS,
        **options,
        "num_predict": calc_result["num_predict"],
        "num_ctx": calc_result["num_ctx"],
    }

    # response_stream = call_ollama_chat(
    #     prompt, model=model, system=system, stream=True, full_stream_response=True, options=options)

    # output = ""
    # for chunk in response_stream:
    #     output += chunk.get("message", {}).get("content", "")

    llm = Ollama(model=model)
    messages = [
        ChatMessage(
            role="system",
            content=system,
        ),
        ChatMessage(role="user", content=prompt),
    ]
    response = llm.chat(messages, options=options)
    output = response.message.content
    token_counts = response.raw['usage']

    return {
        "system": system,
        "prompt": prompt,
        "response": output,
        "llm_settings": {
            "model": model,
            "num_predict": calc_result["num_predict"],
            "num_ctx": calc_result["num_ctx"],
        },
        "token_counts": token_counts
    }


def combine_summaries(summaries: list[SummaryResult], model: str = COMBINE_MODEL, *, system: str = COMBINE_SYSTEM_MESSAGE, depth: int = 1) -> Generator[SummaryResultInfo, None, None]:
    summary_list = [summary['response'] for summary in summaries]

    grouped_summaries = group_summaries(summary_list, model, system=system)

    new_summaries = []

    for group in tqdm(grouped_summaries, desc="Combining summaries", unit="group"):
        prompt = group["summary"]

        summary = generate_summary(prompt, model, system=system)

        yield {"depth": depth, "summary": summary}

        new_summaries.append(summary)

    if len(new_summaries) > 1:
        yield from combine_summaries(
            new_summaries, model, system=system, depth=depth+1)


def summarize_tree(chunks: list[str], model: str = ROOT_MODEL, *, system: str = ROOT_SYSTEM_MESSAGE, combine_model: str = COMBINE_MODEL, combine_system: str = COMBINE_SYSTEM_MESSAGE) -> Generator[SummaryResultInfo, None, None]:
    summaries: list[SummaryResult] = []

    # Initial summarization at the leaf level
    for summary_chunk in tqdm(chunks, desc="Summarizing chunks", unit="chunk"):
        prompt = summary_chunk
        summary = generate_summary(prompt, model, system=system)
        yield {
            "depth": 0,
            "summary": summary
        }

        summaries.append(summary)

    yield from combine_summaries(
        summaries, combine_model, system=combine_system)


def summarize_data(content: str, *, model: str = ROOT_MODEL, system: str = ROOT_SYSTEM_MESSAGE, combine_model: str = COMBINE_MODEL, combine_system: str = COMBINE_SYSTEM_MESSAGE, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> Generator[SummaryResultInfo, None, None] | Generator[SummaryData, None, None]:
    from jet.token.token_utils import get_tokenizer

    tokenizer = get_tokenizer(model)

    buffer_size = overlap
    header_contents = extract_md_header_contents(
        content, tokenizer=tokenizer.encode, max_tokens_per_chunk=chunk_size - buffer_size)
    chunks = []
    for item in header_contents:
        content = item['content']
        chunk_context = ""
        if item['parent_headers']:
            chunk_context = "\n".join(item['parent_headers'])
        chunk = f"{chunk_context}\n{content}"
        chunks.append(chunk)

    logger.info(f"Processing {len(chunks)} chunks")

    summaries: list[SummaryResultInfo] = []
    for summary in summarize_tree(
            chunks, model, system=system,
            combine_model=combine_model, combine_system=combine_system):

        yield summary

        summaries.append(summary)

    yield {
        "chunks": len(chunks),
        "last_depth": summaries[-1]['depth'],
        "final_summary": summaries[-1],
        "other_summaries": summaries[:len(summaries) - 1],
    }

import json
from typing import Any, Generator, List, Optional, TypedDict
from jet.code.splitter_markdown_utils import extract_md_header_contents
from jet.llm.llm_types import OllamaChatOptions
from jet.llm.main.generation import call_ollama_chat
from jet.logger import logger
from jet.token.token_utils import get_tokenizer, token_counter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.ollama.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from tqdm import tqdm

CHUNK_SIZE = 1024
OVERLAP = 128

SETTINGS = {
    "seed": 42,
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.85,
    "stop": None,
    "num_keep": 0,
    "mirostat_tau": 5.0,
}

LOWER_CHAR_SUMMARY_MODEL = "mistral"
ROOT_MODEL = "llama3.2"
COMBINE_MODEL = "mistral"

LOWER_CHAR_SYSTEM_MESSAGE = (
    "You are an AI assistant specialized in summarizing structured content. "
    "Your goal is to generate fewer characters while retaining all key details. "
    "Format the output as structured markdown, maintaining proper headings and bullet points where applicable."
)

ROOT_SYSTEM_MESSAGE = (
    "You are an AI assistant specialized in summarizing unstructured content scraped from the internet. "
    "Your goal is to generate a clear, concise, and factual summary of the given information, "
    "ensuring that no false information is introduced and that all key details are retained. "
    "The summary should always be shorter than the original content, conveying the key points efficiently. "
    "Format the output as structured markdown, maintaining proper headings and bullet points where applicable."
)

COMBINE_SYSTEM_MESSAGE = (
    "You are an AI assistant refining and merging multiple summarized sections into a more concise and coherent summary. "
    "Your goal is to combine the given summaries while eliminating redundancy, ensuring factual accuracy, "
    "and retaining all key details. The final summary should always be shorter than the combined input. "
    "Ensure logical flow and clarity. Format the output in structured markdown where applicable."
)


class TokenCounts(TypedDict):
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


class SummaryResult(TypedDict):
    prompt: str
    response: str
    token_counts: TokenCounts


class SummaryResultInfo(TypedDict):
    depth: int
    summary: SummaryResult


class SummaryData(TypedDict):
    chunks: int
    last_depth: int
    final_summary: SummaryResultInfo
    other_summaries: list[SummaryResultInfo]


def generate_summary(prompt: str, model: str = "llama3.1", system: str = "", options: OllamaChatOptions = {}) -> SummaryResult:
    prompt = prompt.strip()
    system = system.strip()

    prompt_tokens: int = token_counter(prompt, model)
    system_tokens: int = token_counter(system, model)
    prompt_tokens = prompt_tokens + system_tokens

    model_max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[model]
    num_predict = int(prompt_tokens * 0.75)
    num_ctx = prompt_tokens + num_predict

    if num_ctx > model_max_tokens:

        raise ValueError({
            "prompt_tokens": prompt_tokens,
            "num_predict": num_predict,
            "error": f"Context window size ({num_ctx}) exceeds model's maximum tokens ({model_max_tokens})",
        })

    options = {**SETTINGS, **options}
    options["num_predict"] = num_predict
    options["num_ctx"] = num_ctx

    response_stream = call_ollama_chat(
        prompt, model=model, system=system, stream=True, full_stream_response=True, options=options)

    output = ""
    for chunk in response_stream:
        output += chunk.get("message", {}).get("content", "")

    response_tokens: int = token_counter(output, model)
    total_tokens: int = prompt_tokens + response_tokens

    return {
        "prompt": prompt,
        "response": output,
        "token_counts": {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
        }
    }


def generate_combined_summary(summary1: str, summary2: str, model: str = COMBINE_MODEL, *, system: str = COMBINE_SYSTEM_MESSAGE, lower_char_model=LOWER_CHAR_SUMMARY_MODEL, lower_char_system=LOWER_CHAR_SYSTEM_MESSAGE) -> SummaryResult:
    while True:
        if not summary1.startswith("Summary 1:"):
            summary1 = f"Summary 1:\n\n{summary1}"
        if not summary2.startswith("Summary 2:"):
            summary2 = f"Summary 2:\n\n{summary2}"

        combined_text = f"{summary1}\n\n\n{summary2}"
        prompt = combined_text

        try:
            summary = generate_summary(prompt, model, system=system)
            return summary
        except ValueError as e:
            error_data = e.args[0]
            error = error_data.get("error")
            prompt_tokens = error_data.get("prompt_tokens")
            num_predict = error_data.get("num_predict")

            logger.log("Prompt tokens:", prompt_tokens,
                       colors=["GRAY", "WARNING"])
            logger.log("num_predict:", num_predict, colors=["GRAY", "WARNING"])
            logger.warning(error)

            lower_chunk_size = int(prompt_tokens * 0.6)

            final_summary1: str
            final_summary2: str

            # Generate lower summaries for retry
            for result in summarize_data(summary1, chunk_size=lower_chunk_size, model=lower_char_model, system=lower_char_system):
                if "final_summary" in result:
                    final_summary1 = result['final_summary']['summary']['response']

            for result in summarize_data(summary2, chunk_size=lower_chunk_size, model=lower_char_model, system=lower_char_system):
                if "final_summary" in result:
                    final_summary2 = result['final_summary']['summary']['response']

            return generate_combined_summary(final_summary1, final_summary2)


def combine_summaries(summaries: list[SummaryResult], model: str = COMBINE_MODEL, *, system: str = COMBINE_SYSTEM_MESSAGE) -> Generator[SummaryResultInfo, None, None]:
    current_depth = 1

    # Iterative merging of summaries until one remains
    while len(summaries) > 1:
        new_summaries: list[SummaryResult] = []
        for i in range(0, len(summaries), 2):
            if i + 1 < len(summaries):
                summary1 = summaries[i]['response']
                summary2 = summaries[i + 1]['response']
                summary = generate_combined_summary(
                    summary1, summary2, model, system=system)
                yield {
                    "depth": current_depth,
                    "summary": summary
                }
                new_summaries.append(summary)
            else:
                # Carry forward the last remaining summary as is
                new_summaries.append(summaries[i])

        current_depth += 1
        summaries = new_summaries

    yield {
        "depth": current_depth,
        "summary": summaries[0]
    }


def summarize_tree(chunks: List[str], model: str = ROOT_MODEL, *, system: str = ROOT_SYSTEM_MESSAGE, combine_model: str = COMBINE_MODEL, combine_system: str = COMBINE_SYSTEM_MESSAGE) -> Generator[SummaryResultInfo, None, None]:
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

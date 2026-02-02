"""
Agentic RAG over Hugging Face Transformers documentation
(without any LangChain dependencies)

Requirements:
    pip install smolagents rank_bm25 huggingface_hub datasets pyarrow python-dotenv
"""

import re
import shutil
from collections import namedtuple
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.file.utils import save_file
from jet.libs.smolagents.custom_models import OpenAIModel
from rank_bm25 import BM25Okapi
from smolagents import CodeAgent, Tool

load_dotenv()  # loads HF_TOKEN if present in .env

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#   1. Download and prepare knowledge base
# ──────────────────────────────────────────────────────────────────────────────

repo_id = "m-ric/huggingface_doc"
local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["*.csv"],  # or specifically ["huggingface_doc.csv"]
    # cache_dir="./hf_cache",
)

# We only need the csv files — in practice there's usually one or few
csv_files = list(Path(local_dir).rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No .csv files found in {local_dir}")

print(f"Found {len(csv_files)} csv file(s)")

# ──────────────────────────────────────────────────────────────────────────────
#   Very simple document class (replaces langchain Document)
# ──────────────────────────────────────────────────────────────────────────────
Document = namedtuple("Document", ["page_content", "metadata"])


def load_docs_from_csv() -> list[Document]:
    import pandas as pd

    docs = []
    for path in csv_files:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            source = row["source"]
            if not source.startswith("huggingface/transformers"):
                continue
            text = row["text"]
            if not text or len(text.strip()) < 20:
                continue
            # Clean up source if needed — adjust splitting logic depending on actual format
            clean_source = (
                source.split("huggingface/transformers", 1)[-1].lstrip("/")
                if "huggingface/transformers" in source
                else source
            )
            metadata = {"source": clean_source}
            docs.append(Document(page_content=text, metadata=metadata))

    print(f"Loaded {len(docs)} raw documents from transformers section")
    return docs


source_docs = load_docs_from_csv()
save_file(source_docs, f"{OUT_DIR}/source_docs.json")


# ──────────────────────────────────────────────────────────────────────────────
#   Simple recursive character text splitter (no langchain)
# ──────────────────────────────────────────────────────────────────────────────
def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] = ["\n\n", "\n", ".", " ", ""],
    add_start_index: bool = True,
) -> list[Document]:
    chunks = []

    def _split_recursive(txt: str, start_idx: int = 0, depth: int = 0) -> None:
        if depth > 200:  # safety net
            # fallback: force cut without recursion
            pos = 0
            while pos < len(txt):
                end = min(pos + chunk_size, len(txt))
                chunk_text = txt[pos:end]
                chunks.append(
                    Document(
                        page_content=chunk_text.strip(),
                        metadata={"start_index": start_idx + pos}
                        if add_start_index
                        else {},
                    )
                )
                pos += chunk_size - chunk_overlap
            return

        if len(txt) <= chunk_size:
            chunks.append(
                Document(
                    page_content=txt,
                    metadata={"start_index": start_idx} if add_start_index else {},
                )
            )
            return

        split_done = False

        for sep in separators:
            if not sep:
                # ── Empty separator = fallback: iterative chunking ───────────────
                pos = 0
                while pos < len(txt):
                    end = min(pos + chunk_size, len(txt))
                    chunk_text = txt[pos:end]
                    chunks.append(
                        Document(
                            page_content=chunk_text.strip(),
                            metadata={"start_index": start_idx + pos}
                            if add_start_index
                            else {},
                        )
                    )
                    pos += chunk_size - chunk_overlap
                split_done = True
                break  # important: stop after handling fallback

            parts = re.split(f"({re.escape(sep)})", txt)
            if len(parts) <= 1:
                continue  # this separator didn't help → try next

            current = ""
            current_start = start_idx

            for i, part in enumerate(parts):
                if not part:
                    continue

                if len(current) + len(part) <= chunk_size:
                    current += part
                else:
                    if current:
                        chunks.append(
                            Document(
                                page_content=current.strip(),
                                metadata={"start_index": current_start}
                                if add_start_index
                                else {},
                            )
                        )
                    current = part
                    current_start = start_idx + sum(len(p) for p in parts[:i])

            if current:
                chunks.append(
                    Document(
                        page_content=current.strip(),
                        metadata={"start_index": current_start}
                        if add_start_index
                        else {},
                    )
                )

            # Recurse only on the very last piece if it's still too big
            remaining = "".join(parts[-1:]).lstrip(sep)  # avoid leading sep repeat
            if remaining and len(remaining) > chunk_size - chunk_overlap:
                _split_recursive(
                    remaining, start_idx + len(txt) - len(remaining), depth + 1
                )

            split_done = True
            break  # we used this separator → no need to try others

        if not split_done:
            # Should not reach here if "" is in separators
            raise ValueError("No separator worked and empty fallback missing")

    _split_recursive(text)
    return chunks


# Process all documents
docs_processed = []
for doc in source_docs:
    split_docs = recursive_split(
        doc.page_content,
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )
    # attach original source metadata
    for sd in split_docs:
        sd.metadata["source"] = doc.metadata["source"]
    docs_processed.extend(split_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")


# ──────────────────────────────────────────────────────────────────────────────
#   2. BM25 Retriever Tool
# ──────────────────────────────────────────────────────────────────────────────


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses BM25 lexical search to retrieve the most relevant parts of the "
        "Hugging Face Transformers documentation that can help answer the query. "
        "Input should be a short, affirmative search phrase (not a full question)."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query — make it keyword-rich and affirmative",
        }
    }
    output_type = "string"

    def __init__(self, documents: list[Document]):
        super().__init__()
        self.documents = documents

        # Tokenize for BM25 (very simple whitespace + punctuation splitting)
        tokenized_corpus = [
            re.findall(r"\w+|[^\w\s]", doc.page_content.lower()) for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built")

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            return "Error: query must be a string"

        query_tokens = re.findall(r"\w+|[^\w\s]", query.lower())
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-10:][::-1]  # top 10

        retrieved = []
        for i, idx in enumerate(top_indices, 1):
            doc = self.documents[idx]
            header = (
                f"\n\n===== Document {i} (source: {doc.metadata['source']}) =====\n"
            )
            retrieved.append(header + doc.page_content)

        return (
            "Retrieved documents:\n" + "".join(retrieved)
            if retrieved
            else "No relevant documents found."
        )


retriever_tool = RetrieverTool(docs_processed)


# ──────────────────────────────────────────────────────────────────────────────
#   3. Create and run the agent
# ──────────────────────────────────────────────────────────────────────────────


# model = InferenceClientModel()  # default model (currently Qwen-based thinking model)
# or explicitly:
# model = InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 8000,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


model = create_local_model()

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=5,  # increased a bit because BM25 is lexical → may need more iterations
    verbosity_level=2,  # 0 = quiet / 1 = normal / 2 = detailed
)

# Example question
question = "For a transformers model training, which is slower, the forward or the backward pass?"

print("\n" + "=" * 70)
print("Running Agentic RAG on question:")
print(question)
print("=" * 70 + "\n")

final_answer = agent.run(question)

print("\nFinal answer:")
print(final_answer)

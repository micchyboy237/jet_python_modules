# md_summarizer

Recursively summarizes a directory of markdown files using a small, local
llama.cpp model with a limited context window (e.g. 10k tokens). Built for a
setup like: Windows 11, Ryzen 5 3600, GTX 1660 6GB, running `llama-server`
locally.

## How it works

1. **Discover** — walks the target directory recursively, finds every `.md`
   file, and builds a tree mirroring the folder structure.
2. **Map** — each file is summarized into short bullet-point facts (the
   "Mapper" agent). If a file is too big for one call, it's split by markdown
   headers first, then by paragraph if a single header section is still too
   big, and the resulting chunk summaries are merged.
3. **Reduce** — each folder's children (file summaries + any subfolder
   summaries) are merged into one folder-level summary (the "Reducer" agent),
   walking up the tree bottom-up. No embeddings or clustering are needed —
   your existing folder structure already *is* the hierarchy.
4. **Synthesize** — at the root, the top-level folder summaries are combined
   into one coherent prose digest (the "Synthesizer" agent).
5. **Verify (optional)** — a sample of leaf-level facts is spot-checked
   against the final digest (the "Verifier" agent) as a cheap smoke test for
   hallucination drift, since small models compound errors across several
   rounds of merging.

All four roles are the *same* model on your llama.cpp server — they're
distinguished only by system prompt and by where in the tree they run.

## Setup

```bash
pip install requests
```

Start your llama.cpp server with a context size matching what you pass to
`--model-ctx` below (they must agree):

```bash
llama-server -m your-model.Q4_K_M.gguf -c 10000 --host 0.0.0.0 --port 8080
```

## Try it without a server first

```bash
python -m md_summarizer.cli --demo --verbose
```

This runs the full pipeline against the bundled `demo_docs/` (a small fictional
project's docs) using a mock LLM client — no GPU or server required. It proves
the chunking, recursive reduce, and tree-walk logic all work before you point
it at a real model. Watch the log lines: `chunked into N piece(s)`, `level 0 /
level 1 ...`, and `reducing folder ...` show exactly what's happening at each
step.

To see the recursive multi-level reduce actually trigger (rather than
everything fitting in one call), force a tiny budget:

```bash
python -m md_summarizer.cli --demo --model-ctx 220 --reserved-output 40 --prompt-overhead 40 --verbose
```

## Run against your real docs and model

```bash
python -m md_summarizer.cli /path/to/your/docs \
  --server-url http://localhost:8080 \
  --model-ctx 10000 \
  --output summary.md \
  --log-file run.log
```

## Key flags

| Flag | Meaning |
|---|---|
| `--model-ctx` | Must match your `llama-server -c` value |
| `--reserved-output` | Tokens reserved for the model's completion per call |
| `--prompt-overhead` | Tokens reserved for the system prompt + formatting |
| `--no-verify` | Skip the verifier pass (faster, less safe) |
| `--verify-sample-size` | How many leaf facts the verifier spot-checks |
| `--demo` | Use the bundled demo docs + mock client, no server needed |

`input_token_budget = model_ctx - reserved_output - prompt_overhead`. If you
see truncated or garbled output from a real model, the most common fix is
raising `--reserved-output` (the model ran out of room to finish its answer)
or lowering `--prompt-overhead` if your system prompts are actually shorter
than the default estimate.

## Why bullet points between levels, not prose

Small models compound hallucinations across recursive merges. Keeping
intermediate summaries as short, extractive bullet facts (rather than free
prose) makes each merge step easier to verify and harder for the model to
"drift" into inventing connections that aren't in the source. The
Synthesizer only turns things into prose once, at the very end.

## Extending

- Swap `MockLLMClient` for `LlamaCppClient` — done automatically once you
  drop `--demo` and pass `--server-url`.
- Want a different model per role (e.g. a slightly bigger model for the
  Synthesizer)? Give `LLMClient` a second implementation that routes by role,
  or add a `model` field to `PipelineConfig` and thread it through `agents.py`.
- Want incremental runs (only re-summarize changed files)? Cache
  `file_summaries` keyed by `(path, mtime)` between runs — the tree structure
  in `pipeline.py` already gives you a natural place to hang a cache.

## Project layout

```
md_summarizer/
  config.py        # PipelineConfig: token budget math
  llm_client.py     # LlamaCppClient (real) + MockLLMClient (demo)
  chunking.py       # header-aware markdown chunking
  reduce_utils.py   # generic hierarchical_reduce() used by both mapper and reducer
  agents.py         # prompts + mapper/reducer/synthesizer/verifier roles
  pipeline.py       # directory discovery + bottom-up tree processing
  cli.py            # command-line entry point
demo_docs/          # small fictional project used by --demo
```

# smolagents_html_extractor

A modular, testable pipeline to process and extract relevant passages from long HTML documents using `smolagents`-style tools. Includes chunking, extraction, checkpointing, and unit tests.

## Project Structure

```
smolagents_html_extractor/
├── extractor.py          # main pipeline logic
├── tools.py              # custom tools (chunking, extraction, formatting)
├── checkpoint.py         # checkpoint management
├── test_extractor.py     # unit tests
└── requirements.txt
```

## How to Use

### Install dependencies

```sh
pip install -r requirements.txt
```

### Run all tests

```sh
pytest test_extractor.py -v
```

### Example: Extract passages from a URL

```sh
python -c "
from extractor.extractor import run_html_extraction_pipeline
print(run_html_extraction_pipeline(
    'https://example.com/long-page.html',
    'artificial intelligence',
    window_size=3500,
    overlap=700
))
"
```

### Customization

- Adjust `extract_relevant_content()` in `tools.py` for smarter extraction logic, or replace with an LLM call.
- Checkpoints and interim results are stored in the `./checkpoints/` directory.

### Test Coverage

Unit tests exercise chunking, extraction, checkpointing, and the end-to-end pipeline.

---

If you need to (re)generate the project, run:

```sh
bash setup_smolagents_html_extractor.sh
```

Enjoy!

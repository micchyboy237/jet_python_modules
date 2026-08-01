import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
from jet.libs.bertopic.topic_docs_clustering_dynamic import run_bertopic_pipeline
from jet.transformers.object import make_serializable

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


docs = load_sample_jobs_ai_llm_python()
result = run_bertopic_pipeline(
    documents=docs,
    verbose=True,
)

save_file(docs, OUTPUT_DIR / "docs.json")
save_file(make_serializable(result), OUTPUT_DIR / "result.json")

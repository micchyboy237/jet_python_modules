import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.partition.md import partition_md

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

path = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/crawl4ai/docs/md_v2/core"
elements = partition_md(path)

save_file(elements, OUTPUT_DIR / "elements.json")

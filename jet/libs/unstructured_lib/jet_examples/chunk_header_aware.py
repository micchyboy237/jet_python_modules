import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


url = "https://docs.unstructured.io/"
elements = partition_html(url=url)


chunks = chunk_by_title(elements)

save_file(elements, OUTPUT_DIR / "elements.json")
save_file(chunks, OUTPUT_DIR / "chunks.json")

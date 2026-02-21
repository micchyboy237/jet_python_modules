import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

elements = partition_html(url="https://docs.unstructured.io")

save_file(elements, OUTPUT_DIR / "elements.json")

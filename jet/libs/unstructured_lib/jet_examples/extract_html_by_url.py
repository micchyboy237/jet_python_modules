import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# url = "https://docs.unstructured.io"
url = "https://www.onlinejobs.ph/jobseekers/job/1573442"

elements = partition_html(url=url)

save_file(elements, OUTPUT_DIR / "elements.json")

import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_scrape_urls_playwright/missav_ws_en_aed_137/sync_results/page.html"
elements = partition_html(html_file)

save_file(elements, OUTPUT_DIR / "elements.json")

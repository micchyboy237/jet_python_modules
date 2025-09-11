from swarms_tools.search.firecrawl import crawl_entire_site_firecrawl

from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

swarms_ai_content = crawl_entire_site_firecrawl(
    "https://swarms.ai",
    limit=1,
    formats=["markdown"],
    max_wait_time=600,
)

save_file(swarms_ai_content, f"{OUTPUT_DIR}/swarms_ai_content.md")

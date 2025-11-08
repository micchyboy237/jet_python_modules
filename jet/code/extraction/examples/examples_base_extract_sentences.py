from wtpsplit import SaT
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

logger.info("--- Example 1: Basic Usage ---")
# use our '-sm' models for general sentence segmentation tasks
sat_sm = SaT("sat-12l-sm")
sat_sm.half().to("mps") # optional, see above
results = sat_sm.split("this is a test this is another test")
# returns ["this is a test ", "this is another test"]
save_file(results, f"{OUTPUT_DIR}/results.json")

logger.info("--- Example 2: Adapted Style & Domain / Language ---")
# use trained lora modules for strong adaptation to language & domain/style
sat_adapted = SaT("sat-12l-sm", style_or_domain="ud", language="en")
sat_adapted.half().to("mps") # optional, see above
results = sat_adapted.split("This is a test This is another test.")
# returns ['This is a test ', 'This is another test']
save_file(results, f"{OUTPUT_DIR}/results_adapted.json")

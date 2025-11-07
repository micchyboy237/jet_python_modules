from wtpsplit import SaT
from jet.logger import logger
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

logger.info("--- Example 1 ---")
# use our '-sm' models for general sentence segmentation tasks
sat_sm = SaT("sat-12l-sm")
sat_sm.half().to("mps") # optional, see above
results = sat_sm.split("this is a test this is another test")
save_file(results, f"{OUTPUT_DIR}/results.json")
# returns ["this is a test ", "this is another test"]

logger.info("--- Example 2 ---")
# use trained lora modules for strong adaptation to language & domain/style
sat_adapted = SaT("sat-12l-sm", style_or_domain="ud", language="en")
sat_adapted.half().to("mps") # optional, see above
results_adapted = sat_adapted.split("This is a test This is another test.")
save_file(results_adapted, f"{OUTPUT_DIR}/results_adapted.json")
# returns ['This is a test ', 'This is another test']
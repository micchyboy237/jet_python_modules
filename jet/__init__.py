# Initialize all shared modules
# from jet.utils.numpy_config import check_accelerate_usage, check_numpy_config
import logging
import os
import shared.setup.builtin_modules

from transformers import logging as transformers_logging

# from shared.setup.events import setup_events

# Enable parallelism for faster LLM tokenizer encoding

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Set the logging level for transformers to WARNING or higher
transformers_logging.set_verbosity_warning()

# Trigger event initialize
# setup_events()

# check_numpy_config()
# check_accelerate_usage()


def suppress_logging():
    """
    Configure logging to suppress HTTP request logs from urllib3 and requests.
    """
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


suppress_logging()

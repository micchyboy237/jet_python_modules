# Initialize all shared modules
# from jet.utils.inference_config import check_accelerate_usage, check_numpy_config
import shared.setup.builtin_modules

# from shared.setup.events import setup_events

# Enable parallelism for faster LLM tokenizer encoding
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Trigger event initialize
# setup_events()

# check_numpy_config()
# check_accelerate_usage()

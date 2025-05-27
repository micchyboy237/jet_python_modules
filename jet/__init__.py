# Initialize all shared modules
import shared.setup.builtin_modules

# from shared.setup.events import setup_events

# Enable parallelism for faster LLM tokenizer encoding
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Trigger event initialize
# setup_events()

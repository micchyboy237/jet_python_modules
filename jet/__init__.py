# Initialize all shared modules
from .shared_modules import *

from shared.events import setup_events

# Enable parallelism for faster LLM tokenizer encoding
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Trigger event initialize
setup_events()

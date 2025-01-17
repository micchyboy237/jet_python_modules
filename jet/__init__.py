# Setup all shared packages
from .shared_modules import *

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

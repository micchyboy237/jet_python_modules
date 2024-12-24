from .globals import inject_globals

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Ensure globals are injected when the package is imported
inject_globals()

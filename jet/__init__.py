from shared.events import EventSettings
from jet.logger.config import configure_logger

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Trigger event event_pre_start_hook
# EventSettings.pre_start_hook(configure_logger)
EventSettings.pre_start_hook()

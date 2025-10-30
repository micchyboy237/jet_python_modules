import os
from jet.libs.llama_cpp.llamacpp_interceptor import setup_llamacpp_interceptors
from jet.llm.config import DEFAULT_LOG_DIR
from jet.logger import logger, CustomLogger

llamacpp_log_file = f"{DEFAULT_LOG_DIR}/rest.log"

# Ensure log directory exists before logger instantiation
os.makedirs(os.path.dirname(llamacpp_log_file), exist_ok=True)

if os.path.exists(llamacpp_log_file):
    os.remove(llamacpp_log_file)

llamacpp_logger = CustomLogger("rest", filename=llamacpp_log_file)
logger.orange(f"REST logs: {llamacpp_log_file}")

setup_llamacpp_interceptors(logger=llamacpp_logger.log)

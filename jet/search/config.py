import os
from dotenv import load_dotenv, find_dotenv
from jet.logger import logger

ENVS = [
    "CACHE_DIR",
    "CACHE_DURATION",
    "CACHE_FILE",
]


# Clear all environment variables in ENVS
for env in ENVS:
    os.environ.pop(env, None)

# Load cache from .env or defaults
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

default_cache_dir = os.path.join(os.getcwd(), ".cache")
default_cache_duration = 3600
default_cache_file = os.path.join(default_cache_dir, "cache.json")

CACHE_DIR = os.getenv("CACHE_DIR", default_cache_dir)
CACHE_DURATION = int(os.getenv("CACHE_DURATION", default_cache_duration))
CACHE_FILE = os.getenv("CACHE_FILE", default_cache_file)

logger.log("\n====== START CONFIG ======")
for env in ENVS:
    value = eval(env)  # Dynamically evaluate the value of the variable
    logger.log(f"{env}:", value, colors=["GRAY", "DEBUG"])
logger.log("====== END CONFIG ======\n")

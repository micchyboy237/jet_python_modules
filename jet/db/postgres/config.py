from typing import TypedDict
import os
from types import SimpleNamespace


class PostgresConfig(TypedDict):
    DEFAULT_DB: str
    DEFAULT_USER: str
    DEFAULT_PASSWORD: str
    DEFAULT_HOST: str
    DEFAULT_PORT: int


class PostgresConfigObject:
    """Configuration object for PostgreSQL settings with dot notation access."""

    def __init__(self, config: PostgresConfig):
        self.DEFAULT_DB = config["DEFAULT_DB"]
        self.DEFAULT_USER = config["DEFAULT_USER"]
        self.DEFAULT_PASSWORD = config["DEFAULT_PASSWORD"]
        self.DEFAULT_HOST = config["DEFAULT_HOST"]
        self.DEFAULT_PORT = config["DEFAULT_PORT"]


# Define environment groups
ENV_GROUPS = {
    "macbook": PostgresConfig(
        DEFAULT_DB="postgres",
        DEFAULT_USER="jethroestrada",
        DEFAULT_PASSWORD="",
        DEFAULT_HOST="jethros-macbook-air.local",
        DEFAULT_PORT=5432
    ),
    "local": PostgresConfig(
        DEFAULT_DB="postgres",
        DEFAULT_USER="jethroestrada",
        DEFAULT_PASSWORD="",
        DEFAULT_HOST="localhost",
        DEFAULT_PORT=5432
    )
}

# Default configuration if no environment is specified
DEFAULT_CONFIG = ENV_GROUPS["local"]


def load_config() -> PostgresConfigObject:
    """Load PostgreSQL configuration based on JET_POSTGRES_ENV environment variable."""
    env = os.getenv("JET_POSTGRES_ENV", "local")
    if env not in ENV_GROUPS:
        raise ValueError(
            f"Invalid JET_POSTGRES_ENV: {env}. Valid options are {list(ENV_GROUPS.keys())}")
    return PostgresConfigObject(ENV_GROUPS[env])


# Load and export configuration
config = load_config()
DEFAULT_DB = config.DEFAULT_DB
DEFAULT_USER = config.DEFAULT_USER
DEFAULT_PASSWORD = config.DEFAULT_PASSWORD
DEFAULT_HOST = config.DEFAULT_HOST
DEFAULT_PORT = config.DEFAULT_PORT

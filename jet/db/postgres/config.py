import os
from typing import TypedDict


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
    "local": PostgresConfig(
        DEFAULT_DB=os.getenv("DB_POSTGRES_DB_LOCAL", "postgres"),
        DEFAULT_USER=os.getenv("DB_POSTGRES_USER_LOCAL", "jethroestrada"),
        DEFAULT_PASSWORD=os.getenv("DB_POSTGRES_PASSWORD_LOCAL", ""),
        DEFAULT_HOST=os.getenv("DB_POSTGRES_HOST_LOCAL", "localhost"),
        DEFAULT_PORT=int(os.getenv("DB_POSTGRES_PORT_LOCAL", 5432)),
    ),
    "macbook": PostgresConfig(
        DEFAULT_DB=os.getenv("DB_POSTGRES_DB_MAC", "postgres"),
        DEFAULT_USER=os.getenv("DB_POSTGRES_USER_MAC", "jethroestrada"),
        DEFAULT_PASSWORD=os.getenv("DB_POSTGRES_PASSWORD_MAC", ""),
        DEFAULT_HOST=os.getenv("DB_POSTGRES_HOST_MAC", "localhost"),
        DEFAULT_PORT=int(os.getenv("DB_POSTGRES_PORT_MAC", 5432)),
    ),
    "windows": PostgresConfig(
        DEFAULT_DB=os.getenv("DB_POSTGRES_DB_PC", "postgres"),
        DEFAULT_USER=os.getenv("DB_POSTGRES_USER_PC", "jethroestrada"),
        DEFAULT_PASSWORD=os.getenv("DB_POSTGRES_PASSWORD_PC", "1"),
        DEFAULT_HOST=os.getenv("DB_POSTGRES_HOST_PC", "localhost"),
        DEFAULT_PORT=int(os.getenv("DB_POSTGRES_PORT_PC", 5432)),
    ),
}

# Default configuration if no environment is specified
postgres_env = os.getenv("JET_POSTGRES_ENV", "local")
DEFAULT_CONFIG = ENV_GROUPS[postgres_env]


def load_config() -> PostgresConfigObject:
    """Load PostgreSQL configuration based on JET_POSTGRES_ENV environment variable."""
    if postgres_env not in ENV_GROUPS:
        raise ValueError(
            f"Invalid JET_POSTGRES_ENV: {postgres_env}. Valid options are {list(ENV_GROUPS.keys())}"
        )
    return PostgresConfigObject(ENV_GROUPS[postgres_env])


# Load and export configuration
config = load_config()
DEFAULT_DB = config.DEFAULT_DB
DEFAULT_USER = config.DEFAULT_USER
DEFAULT_PASSWORD = config.DEFAULT_PASSWORD
DEFAULT_HOST = config.DEFAULT_HOST
DEFAULT_PORT = config.DEFAULT_PORT

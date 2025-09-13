from jet.logger import logger
from jet._types import ValidationResult


def validate_sql(command: str, read="postgres") -> ValidationResult:
    """
    Validate SQL command syntax using sqlglot.
    Returns True if syntax is valid, False otherwise.
    """
    import sqlglot
    from sqlglot.errors import ParseError
    try:
        result = sqlglot.parse(command, read=read)
        return {"passed": True, "error": None}
    except ParseError as e:
        logger.log("SQL validation error:")
        logger.error(e)
        return {"passed": False, "error": e}


__all__ = [
    "validate_sql",
]

import re
from jet.types import DB_Result
from jet.logger import logger


def log_db_results(results: list[DB_Result]):
    print("\n\n")
    logger.log("---------- Summary ----------")
    logger.info(f"Results ({len(results)})")

    for result_idx, result in enumerate(results):
        print("\n")
        logger.log(f"---------- Command {result_idx + 1} ----------")
        success = result['success']
        command = result['command']
        message = result['message']
        if success:
            logger.debug(command)
            logger.log("Response:")
            logger.success(message)
        else:
            logger.debug(command)
            logger.log("Error:")
            logger.error(message)


def log_sql_results(sql_results):
    success = sql_results['success']
    message = sql_results['message']
    results = sql_results['results']
    db_name = sql_results['db_name']

    logger.log("DB Name:", db_name, colors=["LOG", "DEBUG"])
    if success:
        logger.log("reset_db:", "SUCCESS", colors=["LOG", "SUCCESS"])
    else:
        logger.log("reset_db:", "FAILED", colors=["LOG", "ERROR"])

    log_db_results(results)

    if not success:
        logger.error(message)
        raise Exception(message)


def clean_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from a string.

    Args:
        text (str): The string potentially containing ANSI codes.

    Returns:
        str: A clean string without ANSI codes.
    """
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

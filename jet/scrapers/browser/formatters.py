import re
import subprocess
from datetime import datetime
from typing import List, Optional
from jet.logger import logger


def activate_chrome():
    """
    Activates (focuses) the Google Chrome window on macOS.
    """
    script = 'tell application "Google Chrome" to activate'
    subprocess.run(["osascript", "-e", script])
    logger.log("Activated Google Chrome", colors=["GRAY", "ORANGE"])


def construct_browser_query(
    search_terms: str,
    include_sites: Optional[List[str]] = None,
    exclude_sites: Optional[List[str]] = None,
    after_date: Optional[str] = None,
    before_date: Optional[str] = None
) -> str:
    """
    Constructs a browser search query with options to include/exclude sites and filter by a date range.

    Args:
        search_terms (str): The main search query (e.g., "top 10 romantic comedy anime").
        include_sites (Optional[List[str]]): List of domains to include (e.g., ['myanimelist.net', 'animenewsnetwork.com']).
        exclude_sites (Optional[List[str]]): List of domains to exclude (e.g., ['wikipedia.org', 'imdb.com']).
        after_date (Optional[str]): The date after which results are valid (e.g., '2025-01-01').
        before_date (Optional[str]): The date before which results are valid (e.g., '2025-04-05').

    Returns:
        str: The formatted browser query string.

    Raises:
        ValueError: If `search_terms` is empty, `after_date` or `before_date` are not in correct format, or if sites are not valid.
    """
    if not search_terms:
        raise ValueError("search_terms cannot be empty.")

    # Validate date formats using datetime.strptime
    def validate_date(date_str: Optional[str]):
        if date_str:
            try:
                # Try to parse the date
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD.")

    validate_date(after_date)
    validate_date(before_date)

    # Validate site formats
    if include_sites:
        if not all(re.match(r"^[a-z0-9-]+(?:\.[a-z]{2,})+$", site) for site in include_sites):
            raise ValueError("One or more sites in include_sites are invalid.")

    if exclude_sites:
        if not all(re.match(r"^[a-z0-9-]+(?:\.[a-z]{2,})+$", site) for site in exclude_sites):
            raise ValueError("One or more sites in exclude_sites are invalid.")

    # Start building the query with the search terms
    query = search_terms

    # Add exclusions if provided
    if exclude_sites:
        for site in exclude_sites:
            query += f" -site:{site}"

    # Add inclusions if provided
    if include_sites:
        for site in include_sites:
            query += f" site:{site}"

    # Add date filters if provided
    if after_date:
        query += f" after:{after_date}"
    if before_date:
        query += f" before:{before_date}"

    return query

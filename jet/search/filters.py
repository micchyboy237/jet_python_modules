from datetime import datetime
from typing import List, Dict, Optional


def filter_relevant(results: List[Dict], threshold: float, domain: str = None) -> List[Dict]:
    return [
        result for result in results
        if result["score"] >= threshold and result["url"].startswith("https://") and (domain is None or domain in result["url"])
    ]


def filter_by_date(results: List[Dict], min_date: Optional[datetime] = None) -> List[Dict]:
    """Filters results to include only those published on or before min_date."""
    if min_date is None:
        return results
    filtered_results = []
    for result in results:
        try:
            published_date = result.get("published_date")
            if published_date:
                result_date = datetime.fromisoformat(published_date)
                if result_date >= min_date:
                    filtered_results.append(result)
        except (ValueError, TypeError):
            continue
    return filtered_results


def deduplicate_results(filtered_results: List[Dict]) -> List[Dict]:
    unique_urls = {result["url"]: result for result in filtered_results}
    return list(unique_urls.values())


def sort_by_score(results: List[Dict]) -> List[Dict]:
    return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

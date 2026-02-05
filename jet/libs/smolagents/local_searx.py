# local_searx.py
from typing import Any

import requests


class LocalSearxGoogleSearch:
    """Mimics serpapi.GoogleSearch interface using local SearXNG"""

    def __init__(
        self, params: dict[str, Any], searx_url: str = "http://searxng.local:8888"
    ):
        self.base_url = searx_url.rstrip("/") + "/search"
        self.params = params.copy()

    def get_dict(self) -> dict[str, Any]:  # ← real execution happens here
        if not hasattr(self, "params"):
            raise ValueError("You must call search(params) first")

        q = self.params.get("q", "")
        if not q:
            raise ValueError("Missing search query 'q'")

        searx_params = {
            "q": q,
            "format": "json",
            "categories": "general",  # you can make configurable
            "pageno": self.params.get("start", 0) // 10 + 1,
            "time_range": self._map_time_range(self.params.get("tbs", "")),
        }

        # Optional: engines, language, safesearch, etc.
        if "hl" in self.params:
            searx_params["language"] = self.params["hl"]

        try:
            r = requests.get(self.base_url, params=searx_params, timeout=12)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise RuntimeError(f"SearXNG request failed: {exc}") from exc

        # Try to mimic serpapi structure as much as possible
        results = {
            "search_metadata": {
                "engine": "searxng",
                "query": q,
            },
            "search_information": {
                "total_results": data.get("number_of_results", 0),
                "query_displayed": q,
            },
            "organic_results": [],
        }

        for idx, res in enumerate(data.get("results", []), 1):
            item = {
                "position": idx,
                "title": res.get("title", ""),
                "link": res.get("url", ""),
                "snippet": res.get("content", ""),
            }
            if "publishedDate" in res:
                item["date"] = res["publishedDate"]
            if "source" in res:
                item["source"] = res["source"]

            results["organic_results"].append(item)

        return results

    def _map_time_range(self, tbs: str) -> str | None:
        # Very rough mapping of Google's tbs → SearXNG time_range
        if not tbs or "cdr:1" not in tbs:
            return None
        # you can improve this mapping a lot
        if "cd_min:" in tbs and "cd_max:" in tbs:
            return None  # custom range → not supported easily
        if "qdr:y" in tbs:
            return "year"
        if "qdr:m" in tbs:
            return "month"
        if "qdr:w" in tbs:
            return "week"
        if "qdr:d" in tbs:
            return "day"
        return None

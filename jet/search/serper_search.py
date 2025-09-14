from typing import Any, Dict, List, Optional, TypedDict
import logging

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Literal

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RequestConfig(TypedDict):
    headers: Dict[str, str]
    payload: Dict[str, Any]


class SearchResults(TypedDict):
    answerBox: Optional[Dict[str, Any]]
    knowledgeGraph: Optional[Dict[str, Any]]
    organic: Optional[List[Dict[str, Any]]]
    news: Optional[List[Dict[str, Any]]]
    places: Optional[List[Dict[str, Any]]]
    images: Optional[List[Dict[str, Any]]]


def build_request_config(
    serper_api_key: Optional[str],
    query: str,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    num: Optional[int] = None,
    tbs: Optional[str] = None,
) -> RequestConfig:
    """Build headers and payload for Serper API request."""
    # Load from env if not provided, following LangChain pattern
    effective_key = get_from_dict_or_env(
        {"serper_api_key": serper_api_key}, "serper_api_key", "SERPER_API_KEY")
    headers = {
        "X-API-KEY": effective_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query}
    if gl is not None:
        payload["gl"] = gl
    if hl is not None:
        payload["hl"] = hl
    if num is not None:
        payload["num"] = num
    if tbs is not None:
        payload["tbs"] = tbs
    logger.debug(f"Request config: headers={headers}, payload={payload}")
    return {"headers": headers, "payload": payload}


def google_serper_search(
    query: str,
    search_type: Literal["news", "search", "places", "images"] = "search",
    serper_api_key: Optional[str] = None,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    num: Optional[int] = None,
    tbs: Optional[str] = None,
) -> SearchResults:
    """Perform synchronous Google search via Serper API."""
    config = build_request_config(
        serper_api_key=serper_api_key,
        query=query,
        gl=gl,
        hl=hl,
        num=num,
        tbs=tbs,
    )
    logger.debug(
        f"Sending sync request to https://google.serper.dev/{search_type} with payload={config['payload']}")
    response = requests.post(
        f"https://google.serper.dev/{search_type}",
        headers=config["headers"],
        json=config["payload"],
    )
    logger.debug(
        f"Response status: {response.status_code}, content: {response.text}")
    response.raise_for_status()
    return response.json()


async def agoogle_serper_search(
    query: str,
    search_type: Literal["news", "search", "places", "images"] = "search",
    aiosession: Optional[aiohttp.ClientSession] = None,
    serper_api_key: Optional[str] = None,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    num: Optional[int] = None,
    tbs: Optional[str] = None,
) -> SearchResults:
    """Perform asynchronous Google search via Serper API."""
    config = build_request_config(
        serper_api_key=serper_api_key,
        query=query,
        gl=gl,
        hl=hl,
        num=num,
        tbs=tbs,
    )
    url = f"https://google.serper.dev/{search_type}"
    logger.debug(
        f"Sending async request to {url} with payload={config['payload']}")

    if not aiosession:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=config["payload"],
                headers=config["headers"],
                raise_for_status=False,
            ) as response:
                logger.debug(f"Async response status: {response.status}, content: {await response.text()}")
                return await response.json()
    else:
        async with aiosession.post(
            url,
            json=config["payload"],
            headers=config["headers"],
            raise_for_status=True,
        ) as response:
            logger.debug(f"Async response status: {response.status}, content: {await response.text()}")
            return await response.json()


class GoogleSerperAPIWrapper(BaseModel):
    """Wrapper around the Serper.dev Google Search API.

    You can create a free API key at https://serper.dev.

    To use, you should have the environment variable ``SERPER_API_KEY``
    set with your API key, or pass `serper_api_key` as a named parameter
    to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.utilities import GoogleSerperAPIWrapper
            google_serper = GoogleSerperAPIWrapper()
    """

    k: int = 10
    gl: str = "us"
    hl: str = "en"
    type: Literal["news", "search", "places", "images"] = "search"
    result_key_for_type: Dict[Literal["news", "search", "places", "images"], str] = {
        "news": "news",
        "places": "places",
        "images": "images",
        "search": "organic",
    }
    tbs: Optional[str] = None
    serper_api_key: Optional[str] = None
    aiosession: Optional[aiohttp.ClientSession] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        serper_api_key = get_from_dict_or_env(
            values, "serper_api_key", "SERPER_API_KEY"
        )
        values["serper_api_key"] = serper_api_key
        return values

    def results(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: Optional[int] = None,
        tbs: Optional[str] = None,
    ) -> SearchResults:
        """Run query through GoogleSearch."""
        return google_serper_search(
            query=query,
            gl=gl if gl is not None else self.gl,
            hl=hl if hl is not None else self.hl,
            num=num if num is not None else self.k,
            tbs=tbs if tbs is not None else self.tbs,
            search_type=self.type,
            serper_api_key=self.serper_api_key,
        )

    def run(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: Optional[int] = None,
        tbs: Optional[str] = None,
    ) -> str:
        """Run query through GoogleSearch and parse result."""
        results = google_serper_search(
            query=query,
            gl=gl if gl is not None else self.gl,
            hl=hl if hl is not None else self.hl,
            num=num if num is not None else self.k,
            tbs=tbs if tbs is not None else self.tbs,
            search_type=self.type,
            serper_api_key=self.serper_api_key,
        )
        return self._parse_results(results)

    async def aresults(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: Optional[int] = None,
        tbs: Optional[str] = None,
    ) -> SearchResults:
        """Run query through GoogleSearch."""
        return await agoogle_serper_search(
            query=query,
            gl=gl if gl is not None else self.gl,
            hl=hl if hl is not None else self.hl,
            num=num if num is not None else self.k,
            tbs=tbs if tbs is not None else self.tbs,
            search_type=self.type,
            aiosession=self.aiosession,
            serper_api_key=self.serper_api_key,
        )

    async def arun(
        self,
        query: str,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        num: Optional[int] = None,
        tbs: Optional[str] = None,
    ) -> str:
        """Run query through GoogleSearch and parse result async."""
        results = await agoogle_serper_search(
            query=query,
            gl=gl if gl is not None else self.gl,
            hl=hl if hl is not None else self.hl,
            num=num if num is not None else self.k,
            tbs=tbs if tbs is not None else self.tbs,
            search_type=self.type,
            aiosession=self.aiosession,
            serper_api_key=self.serper_api_key,
        )
        return self._parse_results(results)

    def _parse_snippets(self, results: SearchResults) -> List[str]:
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return [answer_box.get("answer")]
            elif answer_box.get("snippet"):
                return [answer_box.get("snippet").replace("\n", " ")]
            elif answer_box.get("snippetHighlighted"):
                return answer_box.get("snippetHighlighted")

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                snippets.append(f"{title}: {entity_type}.")
            description = kg.get("description")
            if description:
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attribute}: {value}.")

        for result in results[self.result_key_for_type[self.type]][: self.k]:
            if "snippet" in result:
                snippets.append(result["snippet"])
            for attribute, value in result.get("attributes", {}).items():
                snippets.append(f"{attribute}: {value}.")

        if len(snippets) == 0:
            return ["No good Google Search Result was found"]
        return snippets

    def _parse_results(self, results: SearchResults) -> str:
        return " ".join(self._parse_snippets(results))

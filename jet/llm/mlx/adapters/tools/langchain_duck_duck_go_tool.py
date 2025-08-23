# example_duckduckgo_search.py
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun, DDGInput

# Initialize the DuckDuckGoSearchAPIWrapper with custom parameters
api_wrapper = DuckDuckGoSearchAPIWrapper(
    region="wt-wt",  # Worldwide region
    safesearch="moderate",  # Moderate safe search
    time="y",  # Results from the past year
    max_results=5,  # Maximum of 5 results
    source="text"  # Text search
)

# Initialize the DuckDuckGoSearchRun tool
search_tool = DuckDuckGoSearchRun(api_wrapper=api_wrapper)

# Define a search query
query = "latest advancements in AI 2025"

# Run the search synchronously
result = search_tool._run(query)
print("Search Results:")
print(result)

# Example with different parameters (e.g., news source)
news_api_wrapper = DuckDuckGoSearchAPIWrapper(
    region="us-en",  # US English region
    safesearch="off",  # No safe search
    time="m",  # Results from the past month
    max_results=3,  # Maximum of 3 results
    source="news"  # News search
)
news_search_tool = DuckDuckGoSearchRun(api_wrapper=news_api_wrapper)

# Run the news search
news_result = news_search_tool._run(query)
print("\nNews Search Results:")
print(news_result)

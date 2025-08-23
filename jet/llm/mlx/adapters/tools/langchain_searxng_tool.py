# example_searx_search.py
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools.searx_search.tool import SearxSearchRun, SearxSearchQueryInput

# Initialize the SearxSearchWrapper with a self-hosted SearxNG instance
# Replace with your SearxNG instance URL
searx_host = "http://jethros-macbook-air.local:3000"
search_wrapper = SearxSearchWrapper(searx_host=searx_host, unsecure=True)

# Initialize the SearxSearchRun tool
search_tool = SearxSearchRun(wrapper=search_wrapper)

# Define a search query
query = "latest advancements in AI 2025"

# Run the search synchronously
result = search_tool._run(query)
print("Search Results:")
print(result)

# Example with additional parameters (e.g., specific engines)
search_tool_with_params = SearxSearchRun(
    wrapper=SearxSearchWrapper(
        searx_host=searx_host,
        unsecure=True,
        engines=["google", "bing"],
        query_suffix="site:*.edu"  # Restrict to educational sites
    )
)

# Run the search with specific parameters
result_with_params = search_tool_with_params._run(query)
print("\nSearch Results (with engines and suffix):")
print(result_with_params)

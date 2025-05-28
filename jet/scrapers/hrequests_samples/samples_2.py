import hrequests

# Function for Concurrent & Lazy Requests


def concurrent_lazy_requests(urls, nohup=False, size=2):
    """
    Sends requests concurrently to the given list of URLs.
    :param urls: List of URLs to request.
    :param nohup: If True, requests are sent in lazy (background) mode.
    :param size: Number of concurrent requests to send at a time.
    :return: List of responses or LazyResponse objects.
    """
    if nohup:
        responses = hrequests.get(urls, nohup=True)
    else:
        requests = [hrequests.async_get(url) for url in urls]
        responses = hrequests.map(requests, size=size)
    return responses

# Function for HTML Parsing


def parse_html(url, selector, first=True):
    """
    Parses the HTML of the given URL and selects elements based on a CSS selector.
    :param url: URL to fetch and parse HTML from.
    :param selector: CSS selector to find elements.
    :param first: If True, returns the first matching element; otherwise, a list of elements.
    :return: Parsed element(s) matching the CSS selector.
    """
    resp = hrequests.get(url)
    elements = resp.html.find_all(selector)
    if first:
        return elements[0] if elements else None
    return elements

# Sample usage for Concurrent & Lazy Requests


def test_concurrent_lazy_requests():
    urls = ["https://www.python.org",
            "https://github.com/", "https://www.google.com"]
    print("Testing Concurrent Requests...")
    responses = concurrent_lazy_requests(urls, nohup=False, size=3)
    for resp in responses:
        print(f"URL: {resp.url}, Status Code: {resp.status_code}")

    print("\nTesting Lazy Requests...")
    lazy_responses = concurrent_lazy_requests(urls, nohup=True)
    for lazy_resp in lazy_responses:
        print(f"URL: {lazy_resp.url}, Status Code: {lazy_resp.status_code}")

# Sample usage for HTML Parsing


def test_parse_html():
    url = "https://www.python.org"
    print("Testing HTML Parsing...")
    about_section = parse_html(url, "#about", first=True)
    if about_section:
        print(
            f"Element ID: {about_section.attrs.get('id', 'No ID')}, Text: {about_section.text[:100]}...")
    else:
        print("No element found.")


# Run the tests
if __name__ == "__main__":
    test_concurrent_lazy_requests()
    print("\n" + "=" * 50 + "\n")
    test_parse_html()

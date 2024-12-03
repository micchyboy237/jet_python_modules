import hrequests
from hrequests.proxies import evomi

# Function for Browser Automation


def browser_automation():
    """
    Demonstrates browser automation using hrequests.
    """
    # Initialize a BrowserSession
    browser = hrequests.BrowserSession(headless=False, mock_human=True)
    try:
        # Navigate to a website
        browser.goto("https://example.com")
        print("Page Title:", browser.html.title)

        # Interact with the page
        browser.type("input[name='q']", "Hello World!")
        browser.click("input[type='submit']")

        # Take a screenshot
        browser.screenshot(path="example_screenshot.png")
        print("Screenshot saved as 'example_screenshot.png'.")

        # Extract cookies
        cookies = browser.cookies
        print("Cookies extracted:", cookies)
    finally:
        # Close the browser session
        browser.close()

# Function for Evomi Proxies


def evomi_proxies():
    """
    Demonstrates proxy usage with Evomi and hrequests.
    """
    # Initialize a residential proxy
    proxy = evomi.ResidentialProxy(
        username="your_username", key="your_key", country="US", session_type="session")

    # Create a session with the proxy
    session = hrequests.Session(proxy=proxy)
    try:
        # Send a request using the proxy
        response = session.get("https://httpbin.org/ip")
        print("Response Status Code:", response.status_code)
        print("Response Body:", response.text)
    finally:
        # Close the session
        session.close()


# Sample Usage for Browser Automation
print("Running Browser Automation Example...")
browser_automation()

# Sample Usage for Evomi Proxies
print("\nRunning Evomi Proxies Example...")
evomi_proxies()

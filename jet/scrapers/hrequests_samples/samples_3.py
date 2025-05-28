import hrequests
from hrequests.proxies import evomi
import os
import time

# Function for Browser Automation


def browser_automation():
    """
    Demonstrates browser automation using hrequests.
    """
    print("Initializing BrowserSession...")
    try:
        # Initialize a BrowserSession
        browser = hrequests.BrowserSession(headless=False, mock_human=True)
        print("BrowserSession initialized successfully.")
        try:
            # Navigate to a website
            browser.goto("https://example.com")
            # Wait for page to load
            time.sleep(2)  # Simple delay to ensure page loads
            print("Page Title:", browser.html.title or "No title found")

            # Interact with the page (commented out as in original)
            # browser.type("input[name='q']", "Hello World!")
            # browser.click("input[type='submit']")

            # Take a screenshot of the full page
            screenshot_path = "example_screenshot.png"
            browser.screenshot(path=screenshot_path, full_page=True)
            print(f"Screenshot saved as '{screenshot_path}'.")

            # Extract cookies
            cookies = browser.cookies
            print("Cookies extracted:", cookies)
        finally:
            # Close the browser session
            browser.close()
            print("BrowserSession closed.")
    except Exception as e:
        print(f"Error in browser automation: {e}")

# Function for Evomi Proxies


def evomi_proxies(username=None, key=None):
    """
    Demonstrates proxy usage with Evomi and hrequests.
    :param username: Evomi proxy username (optional).
    :param key: Evomi proxy key (optional).
    """
    if not username or not key:
        print("Skipping Evomi proxies: No valid username or key provided.")
        return

    # Initialize a residential proxy
    proxy = evomi.ResidentialProxy(
        username=username, key=key, country="US", session_type="session"
    )

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
        print("Proxy session closed.")


# Sample Usage
if __name__ == "__main__":
    print("Running Browser Automation Example...")
    browser_automation()

    print("\n" + "=" * 50 + "\n")
    print("Running Evomi Proxies Example...")
    # Load credentials from environment variables
    evomi_username = os.getenv("EVOMI_USERNAME", None)
    evomi_key = os.getenv("EVOMI_KEY", None)
    evomi_proxies(username=evomi_username, key=evomi_key)

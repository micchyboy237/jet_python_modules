
from fake_useragent import UserAgent
from scrapy.settings import Settings
from shutil import which

# Configure caching settings
settings = Settings()

# Configure scrapy settings
settings.set('HTTPCACHE_ENABLED', True)
# Cache expires in 1 hour
settings.set('HTTPCACHE_EXPIRATION_SECS', 3600)
settings.set('HTTPCACHE_DIR', '/Users/jethroestrada/Library/Caches/Scrapy')
settings.set('HTTPCACHE_STORAGE',
             'scrapy.extensions.httpcache.FilesystemCacheStorage')
settings.set('HTTPCACHE_GZIP', True)

# Enable Selenium middleware
settings.set('DOWNLOADER_MIDDLEWARES', {
    # 'scrapy_selenium4.SeleniumMiddleware': 800,
    # "jet.scrapers.browser.scrapy_selenium_backup.SeleniumMiddleware": 800,
    "jet.scrapers.browser.scrapy.SeleniumMiddleware": 800,
})

# Configure selenium settings
settings.set('SELENIUM_DRIVER_NAME', 'chrome')
settings.set('SELENIUM_DRIVER_EXECUTABLE_PATH', which('chromedriver'))


# Generate a random User-Agent
ua = UserAgent()
random_user_agent = ua.random

# Define the full list of options
SELENIUM_DRIVER_ARGUMENTS = [
    '--headless=new',  # Comment this out to allow the browser to open
    '--no-sandbox',  # For Linux servers
    '--disable-gpu',  # Disables GPU acceleration (helps in headless mode).
    # Prevents Chrome extensions from loading (reduces detection risk).
    '--disable-extensions',
    # Hides "Chrome is being controlled by automated software" message.
    '--disable-infobars',
    '--disable-dev-shm-usage',  # Prevent crashes on low-memory systems
    # Prevents sites from detecting Selenium automation.
    '--disable-blink-features=AutomationControlled',
    '--incognito',  # Ensure a clean profile for performance
    f'--user-agent={random_user_agent}',  # Randomized User-Agent
]

# Apply settings
settings.set('SELENIUM_DRIVER_ARGUMENTS', SELENIUM_DRIVER_ARGUMENTS)

__all__ = [
    "settings"
]

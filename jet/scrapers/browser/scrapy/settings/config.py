
from scrapy.settings import Settings
from shutil import which

# Configure caching settings
settings = Settings()

# Configure scrapy settings
settings.set('HTTPCACHE_ENABLED', True)
# Cache expires every 12 hours
settings.set('HTTPCACHE_EXPIRATION_SECS', 43200)
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

# SELENIUM_DRIVER_NAME = 'chrome'
# SELENIUM_DRIVER_EXECUTABLE_PATH = which('chromedriver')
# SELENIUM_DRIVER_ARGUMENTS = [
#     '--disable-gpu',
#     '--disable-extensions',
#     '--disable-infobars',
#     '--no-sandbox',
#     '--disable-dev-shm-usage',
#     '--incognito',
#     # '--headless',  # Run Chrome in headless mode
#     # '--headless=new',  # Run Chrome in headless mode
#     '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
# ]

__all__ = [
    "settings"
]

# Playwright executable paths
PLAYWRIGHT_CACHE_DIR = "/Users/jethroestrada/Library/Caches/ms-playwright"

# Specific browser versions
PLAYWRIGHT_CHROMIUM = f"{PLAYWRIGHT_CACHE_DIR}/chromium-1169"
PLAYWRIGHT_FIREFOX = f"{PLAYWRIGHT_CACHE_DIR}/firefox-1475"
PLAYWRIGHT_WEBKIT = f"{PLAYWRIGHT_CACHE_DIR}/webkit-2140"

# Playwright executable paths (full path to executable)
PLAYWRIGHT_CHROMIUM_EXECUTABLE = f"{PLAYWRIGHT_CHROMIUM}/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
PLAYWRIGHT_FIREFOX_EXECUTABLE = f"{PLAYWRIGHT_FIREFOX}/firefox/Nightly.app/Contents/MacOS/firefox"
PLAYWRIGHT_WEBKIT_EXECUTABLE = f"{PLAYWRIGHT_WEBKIT}/pw_run.sh"

# Print the paths to verify
print("Chromium executable path:", PLAYWRIGHT_CHROMIUM_EXECUTABLE)
print("Firefox executable path:", PLAYWRIGHT_FIREFOX_EXECUTABLE)
print("WebKit executable path:", PLAYWRIGHT_WEBKIT_EXECUTABLE)

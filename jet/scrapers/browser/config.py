import os

# Playwright executable paths
PLAYWRIGHT_CACHE_DIR = "/Users/jethroestrada/Library/Caches/ms-playwright"


# Helper function to get latest browser directory dynamically
def get_browser_dir(browser_name: str):
    if not os.path.exists(PLAYWRIGHT_CACHE_DIR):
        return None
    for folder in sorted(os.listdir(PLAYWRIGHT_CACHE_DIR), reverse=True):
        if folder.startswith(browser_name + "-"):
            return os.path.join(PLAYWRIGHT_CACHE_DIR, folder)
    return None


# Specific browser versions (dynamic)
PLAYWRIGHT_CHROMIUM = get_browser_dir("chromium")
PLAYWRIGHT_FIREFOX = get_browser_dir("firefox")
PLAYWRIGHT_WEBKIT = get_browser_dir("webkit")

# === Chromium Executable - Fixed for your installation ===
chromium_base = PLAYWRIGHT_CHROMIUM
PLAYWRIGHT_CHROMIUM_EXECUTABLE = None

if chromium_base:
    possible_paths = [
        # Standard Chromium
        f"{chromium_base}/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        # Apple Silicon Chromium
        f"{chromium_base}/chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
        # Google Chrome for Testing (YOUR CURRENT CASE)
        f"{chromium_base}/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            PLAYWRIGHT_CHROMIUM_EXECUTABLE = path
            break

# Firefox & WebKit
PLAYWRIGHT_FIREFOX_EXECUTABLE = (
    (f"{PLAYWRIGHT_FIREFOX}/firefox/Nightly.app/Contents/MacOS/firefox")
    if PLAYWRIGHT_FIREFOX
    else None
)

PLAYWRIGHT_WEBKIT_EXECUTABLE = (
    (f"{PLAYWRIGHT_WEBKIT}/pw_run.sh") if PLAYWRIGHT_WEBKIT else None
)

# Print the paths to verify
print("Chromium directory:", PLAYWRIGHT_CHROMIUM)
print("Chromium executable path:", PLAYWRIGHT_CHROMIUM_EXECUTABLE)
print("Firefox executable path:", PLAYWRIGHT_FIREFOX_EXECUTABLE)
print("WebKit executable path:", PLAYWRIGHT_WEBKIT_EXECUTABLE)

print(
    "\n✅ Chromium executable exists?",
    os.path.exists(PLAYWRIGHT_CHROMIUM_EXECUTABLE)
    if PLAYWRIGHT_CHROMIUM_EXECUTABLE
    else False,
)


# List of realistic user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.205 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Oracle) Gecko/20100101 Firefox/130.0",
]

USER_AGENT_CONFIGS = [
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.205 Safari/537.36",
        "sec_ch_ua": '"Google Chrome";v="134", "Chromium";v="134", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Chromium";v="134.0.6998.205", "Not:A-Brand";v="24.0.0.0", "Opera";v="119.0.5497.141"',
        "sec_ch_ua_platform": '"macOS"',
        "sec_ch_ua_platform_version": '"14.5.0"',
    },
    {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "sec_ch_ua": '"Google Chrome";v="129", "Chromium";v="129", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Chromium";v="129.0.0.0", "Not:A-Brand";v="24.0.0.0", "Google Chrome";v="129.0.0.0"',
        "sec_ch_ua_platform": '"Windows"',
        "sec_ch_ua_platform_version": '"10.0.0"',
    },
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "sec_ch_ua": '"Safari";v="17", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Safari";v="17.0.0.0", "Not:A-Brand";v="24.0.0.0"',
        "sec_ch_ua_platform": '"macOS"',
        "sec_ch_ua_platform_version": '"14.5.0"',
    },
]

EXTRA_HTTP_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-PH,en-US;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Priority": "u=0, i",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

# # Headers based on the provided sample
# EXTRA_HTTP_HEADERS = {
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#     "Accept-Encoding": "gzip, deflate, br, zstd",
#     "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
#     "Cache-Control": "no-cache",
#     "Pragma": "no-cache",
#     "Priority": "u=0, i",
#     "Sec-Ch-Ua": '"Google Chrome";v="134", "Chromium";v="134", "Not_A Brand";v="24"',
#     "Sec-Ch-Ua-Arch": '"arm"',
#     "Sec-Ch-Ua-Bitness": '"64"',
#     "Sec-Ch-Ua-Full-Version-List": '"Chromium";v="134.0.6998.205", "Not:A-Brand";v="24.0.0.0", "Opera";v="119.0.5497.141"',
#     "Sec-Ch-Ua-Mobile": "?0",
#     "Sec-Ch-Ua-Platform": '"macOS"',
#     "Sec-Ch-Ua-Platform-Version": '"14.5.0"',
#     "Sec-Fetch-Dest": "document",
#     "Sec-Fetch-Mode": "navigate",
#     "Sec-Fetch-Site": "same-origin",
#     "Sec-Fetch-User": "?1",
#     "Upgrade-Insecure-Requests": "1",
#     "Connection": "keep-alive"
# }

HREQUESTS_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-PH,en-US;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Priority": "u=0, i",
    "Sec-Ch-Ua": '"Google Chrome";v="134", "Chromium";v="134", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Arch": '"arm"',
    "Sec-Ch-Ua-Bitness": '"64"',
    "Sec-Ch-Ua-Full-Version-List": '"Chromium";v="134.0.6998.205", "Not:A-Brand";v="24.0.0.0", "Opera";v="119.0.5497.141"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Ch-Ua-Platform-Version": '"14.5.0"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

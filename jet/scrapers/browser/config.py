# Playwright executable paths
PLAYWRIGHT_CACHE_DIR = "/Users/jethroestrada/Library/Caches/ms-playwright"

# Specific browser versions
PLAYWRIGHT_CHROMIUM = f"{PLAYWRIGHT_CACHE_DIR}/chromium-1181"
PLAYWRIGHT_FIREFOX = f"{PLAYWRIGHT_CACHE_DIR}/firefox-1489"
PLAYWRIGHT_WEBKIT = f"{PLAYWRIGHT_CACHE_DIR}/webkit-2191"

# Playwright executable paths (full path to executable)
PLAYWRIGHT_CHROMIUM_EXECUTABLE = f"{PLAYWRIGHT_CHROMIUM}/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
PLAYWRIGHT_FIREFOX_EXECUTABLE = f"{PLAYWRIGHT_FIREFOX}/firefox/Nightly.app/Contents/MacOS/firefox"
PLAYWRIGHT_WEBKIT_EXECUTABLE = f"{PLAYWRIGHT_WEBKIT}/pw_run.sh"

# Print the paths to verify
print("Chromium executable path:", PLAYWRIGHT_CHROMIUM_EXECUTABLE)
print("Firefox executable path:", PLAYWRIGHT_FIREFOX_EXECUTABLE)
print("WebKit executable path:", PLAYWRIGHT_WEBKIT_EXECUTABLE)

# List of realistic user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.205 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Oracle) Gecko/20100101 Firefox/130.0"
]

USER_AGENT_CONFIGS = [
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.205 Safari/537.36",
        "sec_ch_ua": '"Google Chrome";v="134", "Chromium";v="134", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Chromium";v="134.0.6998.205", "Not:A-Brand";v="24.0.0.0", "Opera";v="119.0.5497.141"',
        "sec_ch_ua_platform": '"macOS"',
        "sec_ch_ua_platform_version": '"14.5.0"'
    },
    {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "sec_ch_ua": '"Google Chrome";v="129", "Chromium";v="129", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Chromium";v="129.0.0.0", "Not:A-Brand";v="24.0.0.0", "Google Chrome";v="129.0.0.0"',
        "sec_ch_ua_platform": '"Windows"',
        "sec_ch_ua_platform_version": '"10.0.0"'
    },
    {
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "sec_ch_ua": '"Safari";v="17", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": '"Safari";v="17.0.0.0", "Not:A-Brand";v="24.0.0.0"',
        "sec_ch_ua_platform": '"macOS"',
        "sec_ch_ua_platform_version": '"14.5.0"'
    }
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
    "Connection": "keep-alive"
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
    "Connection": "keep-alive"
}

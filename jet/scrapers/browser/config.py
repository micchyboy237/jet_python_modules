"""
Dynamic, self-validating browser configuration for anti-detection.
All values are derived from the actual runtime environment.
No hardcoded UAs, versions, or platform strings.
"""

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from jet.logger import logger

# ---------------------------------------------------------------------------
# Platform Detection (runtime, never hardcoded)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _detect_platform() -> dict:
    """Detect OS, architecture, and bitness from the live runtime."""
    machine = platform.machine().lower()
    system = platform.system()

    if system == "Darwin":
        os_name = "macOS"
        try:
            result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            os_version = result.stdout.strip()
        except Exception:
            os_version = platform.mac_ver()[0]
    elif system == "Windows":
        os_name = "Windows"
        os_version = platform.version()
    elif system == "Linux":
        os_name = "Linux"
        os_version = platform.release()
    else:
        os_name = system
        os_version = platform.release()

    if machine in ("arm64", "aarch64"):
        arch = "arm"
    elif machine in ("x86_64", "amd64"):
        arch = "x86"
    else:
        arch = machine

    bitness = "64" if sys.maxsize > 2**32 else "32"

    return {
        "os_name": os_name,
        "os_version": os_version,
        "arch": arch,
        "bitness": bitness,
    }


# ---------------------------------------------------------------------------
# Chrome Version Detection
# ---------------------------------------------------------------------------

_SYSTEM_CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/snap/bin/chromium",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


@lru_cache(maxsize=1)
def _find_system_chrome() -> Optional[str]:
    """Locate a working system Chrome binary."""
    for path in _SYSTEM_CHROME_PATHS:
        if os.path.isfile(path):
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and "Chrome" in result.stdout:
                    return path
            except Exception:
                continue
    return None


@lru_cache(maxsize=1)
def _get_chrome_version(chrome_path: str) -> Optional[str]:
    """Extract full version string from a Chrome binary."""
    try:
        result = subprocess.run(
            [chrome_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
        return match.group(1) if match else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Browser Config Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserConfig:
    """Immutable, validated browser configuration."""

    executable_path: Optional[str]
    channel: Optional[str]
    user_agent: str
    sec_ch_ua: str
    sec_ch_ua_full_version_list: str
    sec_ch_ua_platform: str
    sec_ch_ua_platform_version: str
    sec_ch_ua_arch: str
    sec_ch_ua_bitness: str
    sec_ch_ua_mobile: str
    viewport_width: int
    viewport_height: int
    locale: str
    timezone_id: str
    source: str  # "system_chrome" | "playwright_chromium"

    @property
    def extra_http_headers(self) -> dict:
        """Build consistent headers dict ready for Playwright."""
        return {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-PH,en-US;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=0, i",
            "Sec-Ch-Ua": self.sec_ch_ua,
            "Sec-Ch-Ua-Arch": self.sec_ch_ua_arch,
            "Sec-Ch-Ua-Bitness": self.sec_ch_ua_bitness,
            "Sec-Ch-Ua-Full-Version-List": self.sec_ch_ua_full_version_list,
            "Sec-Ch-Ua-Mobile": self.sec_ch_ua_mobile,
            "Sec-Ch-Ua-Platform": self.sec_ch_ua_platform,
            "Sec-Ch-Ua-Platform-Version": self.sec_ch_ua_platform_version,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive",
        }


# ---------------------------------------------------------------------------
# Config Builder with Validation
# ---------------------------------------------------------------------------


def _build_user_agent(version: str, plat: dict) -> str:
    """Construct a UA string consistent with the detected platform."""
    if plat["os_name"] == "macOS":
        os_token = f"Macintosh; Intel Mac OS X {plat['os_version'].replace('.', '_')}"
    elif plat["os_name"] == "Windows":
        os_token = "Windows NT 10.0; Win64; x64"
    else:
        os_token = f"X11; Linux {plat['arch']}"

    return (
        f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{version} Safari/537.36"
    )


def _build_client_hints(version: str, plat: dict) -> dict:
    """Build Client Hints internally consistent with version + platform."""
    major = version.split(".")[0]
    return {
        "sec_ch_ua": f'"Google Chrome";v="{major}", "Chromium";v="{major}", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": (
            f'"Google Chrome";v="{version}", '
            f'"Chromium";v="{version}", '
            f'"Not:A-Brand";v="24.0.0.0"'
        ),
        "sec_ch_ua_platform": f'"{plat["os_name"]}"',
        "sec_ch_ua_platform_version": f'"{plat["os_version"]}"',
        "sec_ch_ua_arch": f'"{plat["arch"]}"',
        "sec_ch_ua_bitness": f'"{plat["bitness"]}"',
        "sec_ch_ua_mobile": "?0",
    }


def _validate_consistency(config: BrowserConfig) -> list[str]:
    """Return list of inconsistency warnings. Empty = clean."""
    issues = []

    ua_match = re.search(r"Chrome/(\d+\.\d+\.\d+\.\d+)", config.user_agent)
    ua_version = ua_match.group(1) if ua_match else None

    hints_match = re.search(
        r'"Google Chrome";v="(\d+\.\d+\.\d+\.\d+)"',
        config.sec_ch_ua_full_version_list,
    )
    hints_version = hints_match.group(1) if hints_match else None

    if ua_version and hints_version and ua_version != hints_version:
        issues.append(
            f"UA version ({ua_version}) != Client Hints version ({hints_version})"
        )

    plat = _detect_platform()
    if f'"{plat["os_name"]}"' not in config.sec_ch_ua_platform:
        issues.append(
            f"Hints platform ({config.sec_ch_ua_platform}) != detected ({plat['os_name']})"
        )

    if f'"{plat["arch"]}"' not in config.sec_ch_ua_arch:
        issues.append(
            f"Hints arch ({config.sec_ch_ua_arch}) != detected ({plat['arch']})"
        )

    return issues


@lru_cache(maxsize=1)
def get_browser_config() -> BrowserConfig:
    """
    Build and validate a browser config. Tries sources in order:
      1. System Chrome (best TLS fingerprint)
      2. Playwright bundled Chromium (fallback)
    Raises RuntimeError if no viable browser found.
    """
    plat = _detect_platform()

    # --- Source 1: System Chrome ---
    chrome_path = _find_system_chrome()
    if chrome_path:
        version = _get_chrome_version(chrome_path)
        if version:
            hints = _build_client_hints(version, plat)
            ua = _build_user_agent(version, plat)

            config = BrowserConfig(
                executable_path=None,
                channel="chrome",
                user_agent=ua,
                viewport_width=1440,
                viewport_height=900,
                locale="en-PH",
                timezone_id="Asia/Manila",
                source="system_chrome",
                **hints,
            )

            issues = _validate_consistency(config)
            if issues:
                for issue in issues:
                    logger.warning(f"Browser config inconsistency: {issue}")
            else:
                logger.success(
                    f"Browser config loaded: system Chrome {version} "
                    f"({plat['os_name']} {plat['arch']})"
                )
            return config

        logger.warning("System Chrome found but version could not be extracted")

    # --- Source 2: Playwright Bundled Chromium ---
    logger.info("System Chrome unavailable, falling back to Playwright Chromium")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            ua_raw = page.evaluate("() => navigator.userAgent")
            version_match = re.search(r"Chrome/(\d+\.\d+\.\d+\.\d+)", ua_raw)
            browser.close()

            if version_match:
                version = version_match.group(1)
                hints = _build_client_hints(version, plat)
                ua = _build_user_agent(version, plat)

                config = BrowserConfig(
                    executable_path=None,
                    channel="chromium",
                    user_agent=ua,
                    viewport_width=1440,
                    viewport_height=900,
                    locale="en-PH",
                    timezone_id="Asia/Manila",
                    source="playwright_chromium",
                    **hints,
                )

                issues = _validate_consistency(config)
                if issues:
                    for issue in issues:
                        logger.warning(f"Fallback config inconsistency: {issue}")
                else:
                    logger.success(
                        f"Browser config loaded: Playwright Chromium {version} "
                        f"({plat['os_name']} {plat['arch']})"
                    )
                return config
    except Exception as e:
        logger.error(f"Playwright Chromium fallback failed: {e}")

    raise RuntimeError(
        "No viable browser found. Install Google Chrome or run "
        "'playwright install chromium'."
    )


# ---------------------------------------------------------------------------
# Backward Compatibility Shims
# ---------------------------------------------------------------------------
# These preserve the old module-level names so that any external code doing
#   from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
# continues to work without modification. Values are derived dynamically
# from get_browser_config() rather than being hardcoded.


def _resolve_executable_path() -> Optional[str]:
    """
    Resolve the executable path for backward compatibility.

    When using system Chrome via channel="chrome", there is no explicit
    executable_path (Playwright resolves it internally). In that case we
    return the discovered system Chrome path so callers that check
    os.path.exists(PLAYWRIGHT_CHROMIUM_EXECUTABLE) still get a valid result.
    """
    try:
        config = get_browser_config()
        # If config has an explicit executable_path, use it
        if config.executable_path:
            return config.executable_path
        # If using system Chrome channel, return the discovered path
        if config.source == "system_chrome":
            return _find_system_chrome()
        # For Playwright bundled Chromium, return None (Playwright resolves internally)
        return None
    except RuntimeError:
        return None


PLAYWRIGHT_CHROMIUM_EXECUTABLE: Optional[str] = _resolve_executable_path()

# Preserve other legacy names that external modules may import
PLAYWRIGHT_CACHE_DIR: str = "/Users/jethroestrada/Library/Caches/ms-playwright"
PLAYWRIGHT_CHROMIUM: Optional[str] = (
    os.path.dirname(PLAYWRIGHT_CHROMIUM_EXECUTABLE)
    if PLAYWRIGHT_CHROMIUM_EXECUTABLE
    else None
)
PLAYWRIGHT_FIREFOX_EXECUTABLE: Optional[str] = None
PLAYWRIGHT_WEBKIT_EXECUTABLE: Optional[str] = None

# Legacy header dicts — now derived from dynamic config for consistency
try:
    _config = get_browser_config()
    EXTRA_HTTP_HEADERS: dict = _config.extra_http_headers
    USER_AGENT_CONFIGS: list[dict] = [
        {
            "user_agent": _config.user_agent,
            "sec_ch_ua": _config.sec_ch_ua,
            "sec_ch_ua_full_version_list": _config.sec_ch_ua_full_version_list,
            "sec_ch_ua_platform": _config.sec_ch_ua_platform,
            "sec_ch_ua_platform_version": _config.sec_ch_ua_platform_version,
        }
    ]
except RuntimeError:
    EXTRA_HTTP_HEADERS = {}
    USER_AGENT_CONFIGS = []

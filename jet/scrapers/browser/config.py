"""
Dynamic, self-validating browser configuration for anti-detection.
All values are derived from the actual runtime environment.
No hardcoded UAs, versions, or platform strings.
Chromium is preferred over Chrome for system browser detection.
Playwright-managed Chromium is dynamically discovered (version-agnostic).
"""

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from jet.logger import logger


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
# Dynamic Playwright Chromium discovery (version-agnostic)
# ---------------------------------------------------------------------------
_PLAYWRIGHT_CACHE_DIR = Path("/Users/jethroestrada/Library/Caches/ms-playwright")


def _find_playwright_chromium() -> Optional[tuple[str, str]]:
    """
    Generically discover the latest Playwright-managed Chromium binary.
    No hardcoded internal paths — recursively finds the actual executable
    regardless of Playwright version, naming convention, or platform.

    Strategy:
      1. Find all chromium-* dirs (exclude headless shell), sort by version desc
      2. For each version dir, recursively locate the browser executable:
         - macOS: find *.app/Contents/MacOS/<binary> where binary != framework helper
         - Windows: find chrome.exe
         - Linux: find 'chrome' executable
      3. Return first valid match

    Returns (path, "playwright_chromium") or None.
    """
    print(f"[PW-TRACE] Starting generic Playwright Chromium discovery")
    print(f"[PW-TRACE] Cache dir: {_PLAYWRIGHT_CACHE_DIR}")
    print(f"[PW-TRACE] Cache dir exists: {_PLAYWRIGHT_CACHE_DIR.is_dir()}")

    if not _PLAYWRIGHT_CACHE_DIR.is_dir():
        logger.warning(f"Playwright cache dir not found: {_PLAYWRIGHT_CACHE_DIR}")
        print(f"[PW-TRACE] ❌ ABORT: Cache directory does not exist")
        return None

    # ── Step 1: Discover and rank chromium version directories ──────────────
    chromium_dirs: list[tuple[int, Path]] = []
    for entry in _PLAYWRIGHT_CACHE_DIR.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("chromium_headless_shell"):
            print(f"[PW-TRACE]   Skipping headless shell: {entry.name}")
            continue
        if not entry.name.startswith("chromium-"):
            continue
        match = re.search(r"chromium-(\d+)", entry.name)
        if match:
            version_num = int(match.group(1))
            chromium_dirs.append((version_num, entry))
            print(f"[PW-TRACE]   ✅ Found chromium-{version_num}")
        else:
            print(f"[PW-TRACE]   ⚠️  Skipped (no version): {entry.name}")

    if not chromium_dirs:
        logger.warning(f"No chromium-* directories in {_PLAYWRIGHT_CACHE_DIR}")
        print(f"[PW-TRACE] ❌ ABORT: No chromium-* directories found")
        return None

    chromium_dirs.sort(key=lambda x: x[0], reverse=True)
    print(f"[PW-TRACE] Versions found (desc): {[v for v, _ in chromium_dirs]}")

    system = platform.system()
    machine = platform.machine().lower()
    print(f"[PW-TRACE] Platform: {system} / {machine}")

    # ── Step 2: Generic executable finder per platform ──────────────────────
    def _find_macos_browser(root: Path) -> Optional[Path]:
        """Find the primary browser binary inside any .app bundle under root."""
        print(f"[PW-TRACE]    [macOS] Scanning for .app bundles under: {root}")
        app_bundles: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            # Prune deep framework/helper dirs to avoid false matches
            rel = dp.relative_to(root)
            if len(rel.parts) > 6:
                dirnames.clear()
                continue
            # Collect .app directories
            for d in list(dirnames):
                if d.endswith(".app"):
                    app_path = dp / d
                    app_bundles.append(app_path)
                    print(f"[PW-TRACE]      Found .app: {app_path.relative_to(root)}")
            # Don't descend into .app bundles via os.walk; we handle them explicitly
            dirnames[:] = [d for d in dirnames if not d.endswith(".app")]

        if not app_bundles:
            print(f"[PW-TRACE]      No .app bundles found")
            return None

        # For each .app, check Contents/MacOS/ for the primary executable
        for app_bundle in app_bundles:
            macos_dir = app_bundle / "Contents" / "MacOS"
            if not macos_dir.is_dir():
                print(f"[PW-TRACE]      Skipping {app_bundle.name}: no Contents/MacOS/")
                continue

            binaries = [f for f in macos_dir.iterdir() if f.is_file()]
            print(
                f"[PW-TRACE]      {app_bundle.name}/Contents/MacOS/ contains: {[b.name for b in binaries]}"
            )

            # Filter out known non-browser helpers
            skip_names = {"crashpad_handler", "gpu-process", "renderer", "broker"}
            candidates = [
                b
                for b in binaries
                if b.name not in skip_names and os.access(b, os.X_OK)
            ]

            if candidates:
                # Prefer the binary whose name matches the .app bundle stem
                app_stem = app_bundle.stem  # e.g., "Google Chrome for Testing"
                exact_match = next((c for c in candidates if c.name == app_stem), None)
                chosen = exact_match or candidates[0]
                print(
                    f"[PW-TRACE]      Selected binary: {chosen.name} (from {len(candidates)} candidate(s))"
                )
                return chosen

        print(f"[PW-TRACE]      No valid browser binary found in any .app")
        return None

    def _find_windows_browser(root: Path) -> Optional[Path]:
        """Find chrome.exe anywhere under root."""
        print(f"[PW-TRACE]    [Windows] Searching for chrome.exe under: {root}")
        for dirpath, _, filenames in os.walk(root):
            if "chrome.exe" in filenames:
                candidate = Path(dirpath) / "chrome.exe"
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    print(f"[PW-TRACE]      Found: {candidate.relative_to(root)}")
                    return candidate
        print(f"[PW-TRACE]      chrome.exe not found")
        return None

    def _find_linux_browser(root: Path) -> Optional[Path]:
        """Find 'chrome' executable anywhere under root."""
        print(f"[PW-TRACE]    [Linux] Searching for 'chrome' under: {root}")
        for dirpath, _, filenames in os.walk(root):
            if "chrome" in filenames:
                candidate = Path(dirpath) / "chrome"
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    print(f"[PW-TRACE]      Found: {candidate.relative_to(root)}")
                    return candidate
        print(f"[PW-TRACE]      'chrome' not found")
        return None

    # Select platform strategy
    if system == "Darwin":
        find_browser = _find_macos_browser
    elif system == "Windows":
        find_browser = _find_windows_browser
    else:
        find_browser = _find_linux_browser

    # ── Step 3: Try each version (newest first) ────────────────────────────
    for version_num, chromium_dir in chromium_dirs:
        print(f"[PW-TRACE] ── Trying chromium-{version_num}: {chromium_dir}")

        try:
            top_level = [e.name for e in chromium_dir.iterdir()]
            print(f"[PW-TRACE]    Contents: {top_level}")
        except Exception as e:
            print(f"[PW-TRACE]    ⚠️  Cannot list contents: {e}")
            continue

        result = find_browser(chromium_dir)
        if result:
            resolved = str(result.resolve())
            logger.info(
                f"Dynamically found Playwright Chromium v{version_num} at: {resolved}"
            )
            print(f"[PW-TRACE] ✅ SUCCESS: Playwright Chromium v{version_num}")
            print(f"[PW-TRACE]    Path: {resolved}")
            return resolved, "playwright_chromium"

        print(f"[PW-TRACE]    ❌ No browser found in chromium-{version_num}")

    logger.warning("No valid Playwright Chromium binary found in any version directory")
    print(f"[PW-TRACE] ❌ FAILED: Exhausted all versions without finding a browser")
    return None


# ---------------------------------------------------------------------------
# System browser paths (fallback only)
# ---------------------------------------------------------------------------
_SYSTEM_BROWSER_PATHS = [
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
    r"C:\Program Files\Chromium\Application\chrome.exe",
    r"C:\Program Files (x86)\Chromium\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


@lru_cache(maxsize=1)
def _find_system_browser() -> Optional[tuple[str, str]]:
    """
    Locate a working browser binary.
    Returns (path, source_label) or None.

    Priority order:
      1. Dynamic Playwright Chromium (latest version in cache)
      2. System Chromium
      3. System Chrome
    """
    # --- Priority 1: Dynamic Playwright Chromium ---
    pw_result = _find_playwright_chromium()
    if pw_result:
        return pw_result

    # --- Priority 2 & 3: System browsers ---
    for path in _SYSTEM_BROWSER_PATHS:
        if not os.path.isfile(path):
            continue
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                continue
            stdout = result.stdout.strip()
            is_chromium_path = "chromium" in path.lower()
            if "Chromium" in stdout and "Chrome" not in stdout.replace("Chromium", ""):
                return path, "system_chromium"
            if "Chrome" in stdout or "Chromium" in stdout:
                label = "system_chromium" if is_chromium_path else "system_chrome"
                return path, label
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def _get_browser_version(browser_path: str) -> Optional[str]:
    """Extract full version string from a Chromium/Chrome binary."""
    try:
        result = subprocess.run(
            [browser_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"(\d+\.\d+\.\d+\.\d+)", result.stdout)
        return match.group(1) if match else None
    except Exception:
        return None


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
    source: str

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
        "sec_ch_ua": f'"Chromium";v="{major}", "Not_A Brand";v="24"',
        "sec_ch_ua_full_version_list": (
            f'"Chromium";v="{version}", "Not:A-Brand";v="24.0.0.0"'
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
        r'"Chromium";v="(\d+\.\d+\.\d+\.\d+)"',
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
      1. Dynamic Playwright Chromium (preferred, version-agnostic)
      2. System Chromium
      3. System Chrome
      4. Playwright bundled Chromium via API (last resort)
    Raises RuntimeError if no viable browser found.
    """
    plat = _detect_platform()
    browser_result = _find_system_browser()

    if browser_result:
        browser_path, source_label = browser_result
        version = _get_browser_version(browser_path)

        if version:
            logger.info(f"Using browser at: {browser_path}")
            hints = _build_client_hints(version, plat)
            ua = _build_user_agent(version, plat)

            # Determine channel based on source
            if source_label in ("playwright_chromium", "system_chromium"):
                channel = "chromium"
            else:
                channel = "chrome"

            config = BrowserConfig(
                executable_path=browser_path,
                channel=channel,
                user_agent=ua,
                viewport_width=1440,
                viewport_height=900,
                locale="en-PH",
                timezone_id="Asia/Manila",
                source=source_label,
                **hints,
            )

            issues = _validate_consistency(config)
            if issues:
                for issue in issues:
                    logger.warning(f"Browser config inconsistency: {issue}")
            else:
                display_name = source_label.replace("_", " ").title()
                logger.success(
                    f"Browser config loaded: {display_name} {version} "
                    f"({plat['os_name']} {plat['arch']})"
                )
            return config

        logger.warning(
            f"Browser found at {browser_path} but version could not be extracted"
        )

    # --- Last resort: Playwright API launch to extract version ---
    logger.info(
        "No system Chromium/Chrome available, falling back to Playwright Chromium"
    )
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
        "No viable browser found. Install Chromium, Google Chrome, or run "
        "'playwright install chromium'."
    )


def _resolve_executable_path() -> Optional[str]:
    """
    Resolve the executable path for backward compatibility.
    When using system browser via channel, there may be an explicit
    executable_path set. For system browsers we return the discovered path
    so callers that check os.path.exists(PLAYWRIGHT_CHROMIUM_EXECUTABLE)
    still get a valid result.
    """
    try:
        config = get_browser_config()
        if config.executable_path:
            return config.executable_path
        if config.source in ("system_chromium", "system_chrome", "playwright_chromium"):
            result = _find_system_browser()
            return result[0] if result else None
        return None
    except RuntimeError:
        return None


PLAYWRIGHT_CHROMIUM_EXECUTABLE: Optional[str] = _resolve_executable_path()
PLAYWRIGHT_CACHE_DIR: str = str(_PLAYWRIGHT_CACHE_DIR)
PLAYWRIGHT_CHROMIUM: Optional[str] = (
    os.path.dirname(PLAYWRIGHT_CHROMIUM_EXECUTABLE)
    if PLAYWRIGHT_CHROMIUM_EXECUTABLE
    else None
)
PLAYWRIGHT_FIREFOX_EXECUTABLE: Optional[str] = None
PLAYWRIGHT_WEBKIT_EXECUTABLE: Optional[str] = None

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

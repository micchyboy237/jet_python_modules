"""
Cloudflare Turnstile challenge detection and handling for Playwright.

Detects Cloudflare interstitial challenge pages and attempts to solve them
by simulating human-like mouse clicks on the Turnstile checkbox.

The Turnstile iframe lives inside a CLOSED shadow root, so standard
page.query_selector() cannot find it. We use multiple strategies:
  1. page.frames — Playwright tracks frames at the protocol level,
     bypassing shadow DOM boundaries entirely.
  2. JS shadow DOM traversal — pierce closed shadow roots via evaluate()
     to locate the iframe element and extract its bounding rect.
  3. Coordinate fallback — use known widget dimensions relative to
     detected challenge container elements.

IMPORTANT: After clicking the checkbox, Cloudflare shows "Verifying..."
and performs server-side validation. The cf_clearance cookie may appear
BEFORE validation completes. If CF rejects the interaction, it reloads
the challenge page. The real success signal is the page navigating away
from "Just a moment..." to the actual destination URL.

Usage:
    from jet.scrapers.browser.cloudflare import handle_cloudflare_challenge
"""

import random
import time

from jet.logger import logger
from playwright.sync_api import Page as SyncPage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CF_TITLE_SIGNALS: list[str] = [
    "just a moment",
    "checking your browser",
    "attention required",
    "verify you are human",
    "additional verification required",
]

_CF_BODY_SIGNALS: list[str] = [
    "verify you are human",
    "additional verification required",
    "checking if the site connection is secure",
    "ray id",
]

# Known Turnstile widget dimensions (used as fallback)
_WIDGET_WIDTH: int = 300
_WIDGET_HEIGHT: int = 65
_CHECKBOX_X_OFFSET: int = 28

_DETECT_POLL_INTERVAL: float = 0.5
_DETECT_MAX_WAIT: int = 10
_DEFAULT_TIMEOUT: int = 30
_DEFAULT_MAX_ATTEMPTS: int = 3

# After clicking, CF shows "Verifying..." for 2-8s before deciding.
# We must wait through this phase before checking results.
_POST_CLICK_VERIFY_WAIT: float = 5.0


# ---------------------------------------------------------------------------
# Detection (silent + logged variants)
# ---------------------------------------------------------------------------


def _find_cf_frame(page: SyncPage):
    """Find Cloudflare frame via page.frames (bypasses closed shadow DOM)."""
    for frame in page.frames:
        if "challenges.cloudflare.com" in frame.url:
            return frame
    return None


def _is_cf_page(page: SyncPage) -> bool:
    """Silent check — no logging. For use inside tight polling loops."""
    try:
        title = page.title().lower()
        if any(signal in title for signal in _CF_TITLE_SIGNALS):
            return True
        if _find_cf_frame(page):
            return True
        body_text = page.inner_text("body").lower()
        if any(signal in body_text for signal in _CF_BODY_SIGNALS):
            return True
        return False
    except Exception:
        return False


def _check_cf_signals(page: SyncPage) -> bool:
    """Logged check — for one-off detection decisions."""
    try:
        title = page.title().lower()
        if any(signal in title for signal in _CF_TITLE_SIGNALS):
            logger.warning(f"Cloudflare challenge detected via title: '{page.title()}'")
            return True
        if _find_cf_frame(page):
            logger.warning("Cloudflare challenge detected via page.frames.")
            return True
        body_text = page.inner_text("body").lower()
        if any(signal in body_text for signal in _CF_BODY_SIGNALS):
            logger.warning("Cloudflare challenge detected via body text.")
            return True
        return False
    except Exception as e:
        logger.debug(f"Error during Cloudflare signal check: {e}")
        return False


def detect_cf_challenge(
    page: SyncPage,
    max_wait: int = _DETECT_MAX_WAIT,
    poll_interval: float = _DETECT_POLL_INTERVAL,
) -> bool:
    """Poll for Cloudflare challenge signals (waits for async widget injection)."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _check_cf_signals(page):
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    logger.debug("No Cloudflare challenge detected — proceeding normally.")
    return False


# ---------------------------------------------------------------------------
# Iframe location — 3 strategies
# ---------------------------------------------------------------------------


def _get_iframe_box_via_frames(page: SyncPage) -> dict | None:
    """Strategy 1: page.frames → frame_element().bounding_box()."""
    cf_frame = _find_cf_frame(page)
    if not cf_frame:
        return None
    try:
        el = cf_frame.frame_element()
        box = el.bounding_box()
        if box and box["width"] > 0 and box["height"] > 0:
            logger.debug(f"Got iframe box via frame_element(): {box}")
            return box
    except Exception as e:
        logger.debug(f"frame_element() bounding_box failed: {e}")
    return None


def _get_iframe_box_via_shadow_pierce(page: SyncPage) -> dict | None:
    """Strategy 2: Pierce closed shadow roots via JS evaluate()."""
    js_code = """
    () => {
        const allElements = document.querySelectorAll('*');
        for (const el of allElements) {
            if (el.shadowRoot) {
                const iframe = el.shadowRoot.querySelector(
                    'iframe[src*="challenges.cloudflare.com"], ' +
                    'iframe[title*="Cloudflare security challenge"]'
                );
                if (iframe) {
                    const rect = iframe.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
                    }
                }
            }
        }
        const directIframe = document.querySelector(
            'iframe[src*="challenges.cloudflare.com"], ' +
            'iframe[title*="Cloudflare security challenge"]'
        );
        if (directIframe) {
            const rect = directIframe.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
            }
        }
        const wrappers = document.querySelectorAll(
            '[class*="turnstile"], [class*="cf-turnstile"], [data-sitekey]'
        );
        for (const w of wrappers) {
            const rect = w.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
            }
        }
        return null;
    }
    """
    try:
        box = page.evaluate(js_code)
        if box:
            logger.debug(f"Got iframe box via shadow pierce: {box}")
            return box
    except Exception as e:
        logger.debug(f"Shadow pierce evaluate failed: {e}")
    return None


def _get_iframe_box_via_page_center(page: SyncPage) -> dict | None:
    """Strategy 3: Estimate widget position from viewport (last resort)."""
    try:
        viewport = page.viewport_size
        if not viewport:
            return None
        est_x = (viewport["width"] - _WIDGET_WIDTH) / 2
        est_y = viewport["height"] * 0.38
        box = {"x": est_x, "y": est_y, "width": _WIDGET_WIDTH, "height": _WIDGET_HEIGHT}
        logger.debug(f"Using estimated iframe box (page center fallback): {box}")
        return box
    except Exception as e:
        logger.debug(f"Page center fallback failed: {e}")
    return None


def _wait_for_iframe_ready(page: SyncPage, timeout: int = 10) -> dict | None:
    """Wait for Turnstile iframe with valid bounding box (3-strategy fallback)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        box = _get_iframe_box_via_frames(page)
        if box:
            return box
        box = _get_iframe_box_via_shadow_pierce(page)
        if box:
            return box
        time.sleep(0.5)

    logger.warning(
        "Could not locate Turnstile iframe via DOM — using coordinate fallback."
    )
    return _get_iframe_box_via_page_center(page)


# ---------------------------------------------------------------------------
# Human-like interaction
# ---------------------------------------------------------------------------


def _human_like_mouse_click(page: SyncPage, x: float, y: float) -> None:
    """Move mouse with jitter from random offset, then click."""
    start_x = x + random.uniform(-120, 120)
    start_y = y + random.uniform(-60, 60)

    page.mouse.move(start_x, start_y)
    time.sleep(random.uniform(0.1, 0.3))

    steps = random.randint(10, 20)
    for i in range(1, steps + 1):
        ratio = i / steps
        cx = start_x + (x - start_x) * ratio + random.uniform(-2, 2)
        cy = start_y + (y - start_y) * ratio + random.uniform(-2, 2)
        page.mouse.move(cx, cy)
        time.sleep(random.uniform(0.01, 0.04))

    time.sleep(random.uniform(0.05, 0.15))
    page.mouse.click(x, y)
    logger.debug(f"Human-like click at ({x:.0f}, {y:.0f})")


# ---------------------------------------------------------------------------
# Post-click verification
# ---------------------------------------------------------------------------


def _has_clearance_cookie(page: SyncPage) -> bool:
    """Check if cf_clearance cookie exists."""
    try:
        cookies = page.context.cookies()
        return any(c["name"] == "cf_clearance" for c in cookies)
    except Exception:
        return False


def _wait_for_challenge_resolved(
    page: SyncPage,
    original_url: str,
    timeout: int = 25,
) -> bool:
    """
    Wait for the Cloudflare challenge to actually resolve after clicking.

    After clicking the checkbox, Cloudflare goes through phases:
      1. "Verifying..." spinner (2-8 seconds)
      2a. Success → redirects to real page (title changes, URL may change)
      2b. Failure → reloads "Just a moment..." challenge page

    The cf_clearance cookie can appear during phase 1 BEFORE the server
    decides. So we ignore the cookie and instead watch for:
      - Title no longer contains CF signals (= real page loaded)
      - URL changed from the challenge URL

    Uses silent _is_cf_page() to avoid log spam.

    Args:
        page: Playwright sync Page.
        original_url: The URL we navigated to (to detect URL change).
        timeout: Max seconds to wait.

    Returns:
        True if challenge resolved (real page loaded).
        False if still stuck or looped back to challenge.
    """
    deadline = time.time() + timeout
    last_log_time = 0.0

    while time.time() < deadline:
        # Let CF's "Verifying..." phase play out
        try:
            page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass

        # Real success: page is no longer a CF challenge page
        if not _is_cf_page(page):
            logger.debug("Challenge resolved — real page loaded.")
            return True

        # Throttled logging (every ~3s)
        now = time.time()
        if now - last_log_time >= 3.0:
            remaining = deadline - now
            logger.debug(
                f"Waiting for CF challenge to resolve... ({remaining:.0f}s remaining)"
            )
            last_log_time = now

        time.sleep(0.5)

    return False


# ---------------------------------------------------------------------------
# Challenge solver
# ---------------------------------------------------------------------------


def handle_cloudflare_challenge(
    page: SyncPage,
    timeout: int = _DEFAULT_TIMEOUT,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """
    Detect and attempt to solve a Cloudflare Turnstile challenge.

    Critical behavior after clicking:
      - CF shows "Verifying..." for 2-8s while validating server-side
      - cf_clearance cookie may appear BEFORE validation completes
      - If CF rejects the click, it RELOADS the challenge page
      - Real success = title changes away from "Just a moment..."

    Per-attempt flow:
      1. Wait for widget to render (randomized delay mimics human reading)
      2. Locate Turnstile iframe (3 fallback strategies)
      3. Human-like mouse click on checkbox
      4. Wait through "Verifying..." phase (_POST_CLICK_VERIFY_WAIT)
      5. Poll for real page load (title changed, no more CF signals)
      6. If still on challenge → click was rejected → retry

    Returns:
        True if passed (or no challenge present). False if unsolvable.
    """
    if not detect_cf_challenge(page):
        return True

    logger.info("Attempting to solve Cloudflare Turnstile challenge...")
    original_url = page.url

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Cloudflare challenge attempt {attempt}/{max_attempts}")

        # --- Pre-click delay ---
        # Humans don't click instantly. CF measures time-from-page-load-to-click.
        # Longer, randomized delays reduce bot detection. [[4]]
        pre_click_delay = random.uniform(3.0, 6.0)
        logger.debug(f"Pre-click human delay: {pre_click_delay:.1f}s")
        time.sleep(pre_click_delay)

        # --- Locate iframe ---
        box = _wait_for_iframe_ready(page, timeout=10)
        if not box:
            logger.warning(
                f"Attempt {attempt}/{max_attempts}: Could not locate Turnstile widget."
            )
            time.sleep(random.uniform(2.0, 4.0))
            continue

        # --- Click ---
        click_x = box["x"] + _CHECKBOX_X_OFFSET
        click_y = box["y"] + box["height"] / 2
        logger.info(
            f"Turnstile widget: x={box['x']:.0f} y={box['y']:.0f} "
            f"w={box['width']:.0f} h={box['height']:.0f} "
            f"-> clicking ({click_x:.0f}, {click_y:.0f})"
        )
        _human_like_mouse_click(page, click_x, click_y)

        # --- Post-click: wait through "Verifying..." phase ---
        # CF takes 2-8s to validate. Don't check anything during this window.
        verify_wait = _POST_CLICK_VERIFY_WAIT + random.uniform(0, 3)
        logger.debug(f"Waiting {verify_wait:.1f}s for CF 'Verifying...' phase...")
        time.sleep(verify_wait)

        # --- Check result ---
        has_cookie = _has_clearance_cookie(page)
        if has_cookie:
            logger.debug("cf_clearance cookie detected.")

        # The real test: did the page navigate away from the challenge?
        resolved = _wait_for_challenge_resolved(page, original_url, timeout=5)

        if resolved:
            logger.success("✅ Cloudflare challenge passed — redirected to real page.")
            time.sleep(random.uniform(1.0, 2.0))
            return True

        # Still on challenge page after waiting = click was rejected
        if has_cookie:
            logger.warning(
                f"Attempt {attempt}/{max_attempts}: cf_clearance cookie exists "
                "but page still shows challenge — CF rejected the click. "
                "Clearing cookies and retrying..."
            )
            try:
                page.context.clear_cookies()
            except Exception:
                pass
        else:
            logger.warning(
                f"Attempt {attempt}/{max_attempts}: No redirect after click. "
                "Retrying..."
            )

        # Inter-attempt delay — longer pause to avoid rapid-fire pattern [[4]]
        retry_delay = random.uniform(3.0, 6.0)
        logger.debug(f"Inter-attempt delay: {retry_delay:.1f}s")
        time.sleep(retry_delay)

    logger.error(
        f"❌ Failed to solve Cloudflare challenge after {max_attempts} attempts. "
        "Consider integrating a CAPTCHA solver service (2Captcha / CapSolver)."
    )
    return False

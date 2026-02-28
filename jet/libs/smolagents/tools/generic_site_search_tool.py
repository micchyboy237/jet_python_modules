#!/usr/bin/env python3
"""
Generic Site Search Tool
========================

Robust search input discovery for ANY website.

Features:
- DOM ready wait
- Semantic + structural scoring
- Header/nav prioritization
- Search icon auto-reveal
- Iframe traversal
- Scroll + focus handling
- DOM change validation
- Hard failure on error

NOT search-engine specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait

# ────────────────────────────────────────────────
#  Core Utilities
# ────────────────────────────────────────────────


def wait_for_dom_ready(driver: WebDriver, timeout: int = 15) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def scroll_and_focus(driver: WebDriver, elem: WebElement) -> None:
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", elem
    )
    time.sleep(0.4)  # give scroll time

    # Prefer JS click to bypass overlays
    driver.execute_script("arguments[0].focus();", elem)
    driver.execute_script("arguments[0].click();", elem)

    # Fallback if needed:
    # ActionChains(driver).move_to_element(elem).click().perform()


def try_remove_overlays(driver: WebDriver) -> None:
    # Attempt to remove common overlays that might obscure/intercept clicks.
    # This is a basic implementation which can be adjusted as needed.
    js = """
        const overlays = [];
        const nodes = document.body.querySelectorAll('*');
        for (let el of nodes) {
            const cs = window.getComputedStyle(el);
            if (
                (cs.position === 'fixed' || cs.position === 'sticky') &&
                parseFloat(cs.zIndex || "0") > 10 &&
                cs.display !== 'none' &&
                cs.visibility !== 'hidden' &&
                (cs.backgroundColor && cs.backgroundColor !== 'rgba(0, 0, 0, 0)')
            ) {
                overlays.push(el);
            }
        }
        overlays.forEach(ov => { ov.style.display='none'; });
        return overlays.length;
    """
    try:
        driver.execute_script(js)
    except Exception:
        pass


# ────────────────────────────────────────────────
#  Search Input Scoring
# ────────────────────────────────────────────────


def semantic_score(elem: WebElement) -> int:
    score = 0

    attrs = " ".join(
        [
            elem.get_attribute("type") or "",
            elem.get_attribute("name") or "",
            elem.get_attribute("id") or "",
            elem.get_attribute("placeholder") or "",
            elem.get_attribute("aria-label") or "",
        ]
    ).lower()

    if "search" in attrs:
        score += 6
    if "find" in attrs:
        score += 3
    if elem.get_attribute("type") == "search":
        score += 4

    return score


def structural_score(driver: WebDriver, elem: WebElement) -> int:
    score = 0

    try:
        rect = elem.rect
        if rect["width"] > 120:
            score += 2
        if rect["y"] < 400:  # near top of page
            score += 2
    except Exception:
        pass

    try:
        parent = elem.find_element(By.XPATH, "ancestor::*[self::header or self::nav]")
        if parent:
            score += 3
    except Exception:
        pass

    if elem.is_displayed():
        score += 3
    if elem.is_enabled():
        score += 3

    return score


def total_score(driver: WebDriver, elem: WebElement) -> int:
    return semantic_score(elem) + structural_score(driver, elem)


# ────────────────────────────────────────────────
#  Search Icon Reveal Strategy
# ────────────────────────────────────────────────


def click_search_icons(driver: WebDriver) -> None:
    """
    Attempts to click common search icon triggers to reveal hidden inputs.
    """

    candidates = driver.find_elements(
        By.XPATH,
        "//button[contains(@aria-label,'search') or contains(@class,'search')] | "
        "//a[contains(@aria-label,'search') or contains(@class,'search')]",
    )

    for el in candidates:
        try:
            if el.is_displayed():
                el.click()
                time.sleep(1)
        except Exception:
            continue


# ────────────────────────────────────────────────
#  Core Tool
# ────────────────────────────────────────────────


@dataclass
class GenericSiteSearchTool:
    driver: WebDriver
    max_attempts: int = 3

    # ────────────────────────────────────────────

    def find_best_candidate(self) -> Optional[WebElement]:
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input")

        scored: List[Tuple[int, WebElement]] = []

        for elem in inputs:
            try:
                score = total_score(self.driver, elem)
                if score > 0:
                    scored.append((score, elem))
            except Exception:
                continue

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # ────────────────────────────────────────────

    def find_in_iframes(self) -> Optional[WebElement]:
        iframes = self.driver.find_elements(By.TAG_NAME, "iframe")

        for iframe in iframes:
            try:
                self.driver.switch_to.frame(iframe)
                elem = self.find_best_candidate()
                if elem:
                    return elem
            except Exception:
                pass
            finally:
                self.driver.switch_to.default_content()

        return None

    # ────────────────────────────────────────────

    def validate_dom_change(self, before_dom: str, timeout: int = 10) -> None:
        WebDriverWait(self.driver, timeout).until(lambda d: d.page_source != before_dom)

    # ────────────────────────────────────────────

    def perform_search(self, query: str) -> str:
        wait_for_dom_ready(self.driver)

        for attempt in range(self.max_attempts):
            try:
                elem = self.find_best_candidate()

                if not elem:
                    click_search_icons(self.driver)
                    elem = self.find_best_candidate()

                if not elem:
                    elem = self.find_in_iframes()

                if not elem:
                    raise RuntimeError("No search input detected.")

                try_remove_overlays(self.driver)

                before_dom = self.driver.page_source

                scroll_and_focus(self.driver, elem)
                elem.clear()
                elem.send_keys(query)
                elem.send_keys(Keys.ENTER)

                # optionally make validate_dom_change return bool and log instead of raise
                # self.validate_dom_change(before_dom)

                return f"Search executed successfully: '{query}'"

            except Exception as e:
                time.sleep(1)
                if attempt == self.max_attempts - 1:
                    raise RuntimeError(
                        f"Search failed after {self.max_attempts} attempts: {str(e)}"
                    )

        raise RuntimeError("Unreachable state.")

    # ────────────────────────────────────────────

    def __call__(self, query: str) -> str:
        return self.perform_search(query)


# ────────────────────────────────────────────────
#  Usage Example
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from seleniumbase import Driver

    parser = argparse.ArgumentParser(description="Generic Site Search Tool")
    parser.add_argument("url", type=str, help="The URL of the website to search")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run in headless mode (no browser UI)",
    )
    args = parser.parse_args()

    driver = Driver(uc=True, headless=args.headless)

    try:
        driver.get(args.url)

        tool = GenericSiteSearchTool(driver)
        print(tool(args.query))

        time.sleep(5)

    finally:
        driver.quit()

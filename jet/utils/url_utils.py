import requests
import xml.etree.ElementTree as ET
import re


def get_sitemap_url_from_robots(base_url: str) -> str | None:
    try:
        robots_url = f"{base_url.rstrip('/')}/robots.txt"
        res = requests.get(robots_url, timeout=5)
        res.raise_for_status()
        match = re.search(r"Sitemap:\s*(.+)", res.text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch robots.txt: {e}")
        return None


def parse_sitemap_recursive(url: str, collected: set = None) -> list:
    collected = collected or set()

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        root = ET.fromstring(res.content)

        # XML namespaces
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        for sitemap in root.findall("ns:sitemap", ns):
            loc = sitemap.find("ns:loc", ns)
            if loc is not None:
                parse_sitemap_recursive(loc.text.strip(), collected)

        for url_entry in root.findall("ns:url", ns):
            loc = url_entry.find("ns:loc", ns)
            if loc is not None:
                collected.add(loc.text.strip())

    except Exception as e:
        print(f"[ERROR] Failed to parse sitemap {url}: {e}")

    return sorted(collected)


def get_all_sitemap_urls(base_url: str) -> list:
    root_sitemap_url = get_sitemap_url_from_robots(base_url)
    if not root_sitemap_url:
        print("[WARN] No sitemap found in robots.txt")
        return []

    print(f"[INFO] Root sitemap: {root_sitemap_url}")
    return parse_sitemap_recursive(root_sitemap_url)


# Example
if __name__ == "__main__":
    all_urls = get_all_sitemap_urls("https://sitegiant.ph")
    print(f"\nFound {len(all_urls)} URLs:")
    for url in all_urls[:10]:  # just print first 10 for brevity
        print("-", url)

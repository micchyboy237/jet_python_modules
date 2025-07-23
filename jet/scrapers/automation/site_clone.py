import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from jet.file.utils import save_file


def fetch_url(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def download_file(url, root_folder):
    parsed = urlparse(url)
    path = parsed.path.lstrip('/')
    local_path = os.path.join(root_folder, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(local_path, 'wb') as f:
        f.write(resp.content)
    return local_path


def rewrite_links(soup, tag, attr, base_url, root_folder):
    for el in soup.find_all(tag):
        link = el.get(attr)
        if not link:
            continue
        full = urljoin(base_url, link)
        if urlparse(full).netloc != urlparse(base_url).netloc:
            continue
        local = download_file(full, root_folder)
        rel = os.path.relpath(local, root_folder)
        el[attr] = rel


def clone_site(start_url, out_folder='generated/mirror'):
    os.makedirs(out_folder, exist_ok=True)
    html = fetch_url(start_url)
    soup = BeautifulSoup(html, 'html.parser')

    rewrite_links(soup, 'link', 'href', start_url, out_folder)
    rewrite_links(soup, 'script', 'src', start_url, out_folder)
    rewrite_links(soup, 'img', 'src', start_url, out_folder)

    save_file(soup.prettify(), os.path.join(out_folder, 'index.html'))

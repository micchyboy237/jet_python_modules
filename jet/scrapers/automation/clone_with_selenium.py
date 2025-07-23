from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os


def clone_after_render(url, out_folder='selenium_mirror'):
    os.makedirs(out_folder, exist_ok=True)
    opts = Options()
    opts.add_argument('--headless')
    driver = webdriver.Chrome(options=opts)
    driver.get(url)
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, 'html.parser')

    # Similar rewrite_links usage here...
    with open(os.path.join(out_folder, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(soup.prettify())

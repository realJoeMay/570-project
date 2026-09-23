from pathlib import Path
from hashlib import sha256
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup





def get_page(url: str, cache_dir: Path = Path("data/cache/html")) -> tuple[bytes, str]:
    """Read cached HTML, or download and cache a successful response."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256(url.encode("utf-8")).hexdigest()
    html_path = cache_dir / f"{cache_key}.html"
    url_path = cache_dir / f"{cache_key}.url"

    if html_path.is_file():
        # Preserve the final URL after redirects for relative website links.
        page_url = url_path.read_text(encoding="utf-8") if url_path.is_file() else url
        return html_path.read_bytes(), page_url

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    url_path.write_text(response.url, encoding="utf-8")
    # Rename only after the complete HTML has been written.
    temporary_path = html_path.with_suffix(".html.tmp")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(html_path)
    return response.content, response.url


def scrape_parishes(diocese: str, url: str) -> pd.DataFrame:
    html, page_url = get_page(url)
    soup = BeautifulSoup(html, "html.parser")
    parishes = []

    for parish in soup.select("li.site .siteInfo"):
        name_element = parish.select_one(".title .name")
        if name_element is None or not name_element.get_text(strip=True):
            raise ValueError(f"Missing directory entry name for {diocese}: {url}")
        address_element = parish.select_one(".title .address")
        website = parish.select_one(".website a[href]")
        category = parish.find_parent("li", class_="category")
        category_name = category.select_one(".categoryName") if category else None
        parishes.append({
            "name": name_element.get_text(" ", strip=True),
            "address": address_element.get_text(" ", strip=True) if address_element else "",
            "site_url": urljoin(page_url, website["href"]) if website else "",
            "diocese": diocese,
            "category": category_name.get_text(" ", strip=True) if category_name else "",
            "parish_list_url": url,
        })

    if not parishes:
        raise ValueError(f"No parish directory entries found for {diocese}: {url}")
    return pd.DataFrame(parishes).drop_duplicates().reset_index(drop=True)

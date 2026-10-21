import os
import re
from pathlib import Path
from hashlib import sha256
from urllib.parse import urljoin, urlparse
import time
import random
import urllib.robotparser as robotparser
from typing import Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from bs4 import BeautifulSoup
from dotenv import dotenv_values


PARISH_COLUMNS = ["name", "address", "site_url", "diocese", "category", "parish_list_url"]


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve an explicit folder, HTML_CACHE_DIR, .env, or the local default."""
    if cache_dir is None:
        cache_dir = (
            os.environ.get("HTML_CACHE_DIR")
            or dotenv_values(".env").get("HTML_CACHE_DIR")
            or "data/cache/html"
        )
    return Path(cache_dir).expanduser()


def get_page(
    url: str,
    cache_dir: str | Path | None = None,
    *,
    fetch_sources: list[str] | None = None,
    cache_only: bool = False,
) -> tuple[bytes, str]:
    """Read cached HTML, or download and cache a successful response.

    Optionally record whether each fetch uses the cache or the web.
    With cache_only=True, raise FileNotFoundError rather than downloading.
    """
    cache_dir = get_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256(url.encode("utf-8")).hexdigest()
    html_path = cache_dir / f"{cache_key}.html"
    url_path = cache_dir / f"{cache_key}.url"

    if html_path.is_file():
        if fetch_sources is not None:
            fetch_sources.append("cache")
        # Preserve the final URL after redirects for relative website links.
        page_url = url_path.read_text(encoding="utf-8") if url_path.is_file() else url
        return html_path.read_bytes(), page_url

    if cache_only:
        raise FileNotFoundError(f"No cached HTML for {url}")

    if fetch_sources is not None:
        fetch_sources.append("web")
    # Use a Session with a sane User-Agent and retries to reduce bot-blocking
    session = requests.Session()
    default_ua = (
        os.environ.get("SCRAPER_USER_AGENT")
        or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/120.0.0.0 Safari/537.36"
    )
    session.headers.update(
        {
            "User-Agent": default_ua,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    # Retries for transient server errors and rate limits (not for 403).
    retries = Retry(
        total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Respect robots.txt and per-site crawl delays.
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or "/"

    # Module-level cache of RobotFileParser and last fetch times
    global _robots_parsers  # declared below
    global _last_fetch_times
    try:
        parser = _robots_parsers.get(origin)
    except NameError:
        _robots_parsers = {}
        parser = None
    try:
        _ = _last_fetch_times
    except NameError:
        _last_fetch_times = {}

    if parser is None:
        parser = robotparser.RobotFileParser()
        parser.set_url(urljoin(origin, "/robots.txt"))
        try:
            # Use the same User-Agent as page requests. Some sites reject the
            # default urllib agent even though their published rules allow us.
            robots_response = session.get(parser.url, timeout=30)
            if robots_response.status_code in {401, 403}:
                parser.disallow_all = True
            elif 400 <= robots_response.status_code < 500:
                parser.allow_all = True
            else:
                robots_response.raise_for_status()
                parser.parse(robots_response.text.splitlines())
        except requests.RequestException:
            pass
        _robots_parsers[origin] = parser

    # Determine UA for robots checks (RobotFileParser expects a user-agent token)
    ua_for_robots = session.headers.get("User-Agent", "*")
    try:
        allowed = parser.can_fetch(ua_for_robots, path)
    except Exception:
        allowed = True
    # Check environment flags to allow Selenium fallback or ignore robots
    allow_selenium = os.environ.get("SCRAPER_USE_SELENIUM", "").lower() in (
        "1",
        "true",
        "yes",
    )
    ignore_robots = os.environ.get("SCRAPER_IGNORE_ROBOTS", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if not allowed and not (allow_selenium or ignore_robots):
        raise requests.exceptions.HTTPError(
            f"Fetching disallowed by robots.txt for {url} (User-Agent: {ua_for_robots})"
        )

    # Respect crawl-delay if provided, otherwise use randomized polite delay
    crawl_delay = None
    try:
        crawl_delay = parser.crawl_delay(ua_for_robots)
    except Exception:
        crawl_delay = None

    wait = float(crawl_delay) if crawl_delay is not None else random.uniform(1.0, 3.0)
    last = _last_fetch_times.get(origin, 0)
    elapsed = time.time() - last
    if elapsed < wait:
        time.sleep(wait - elapsed)

    response = None
    try:
        response = session.get(url, timeout=30)
        _last_fetch_times[origin] = time.time()
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # If blocked with 403 and Selenium is allowed, attempt a headless browser fetch
        if response is not None and response.status_code == 403 and allow_selenium:
            try:
                content, final = _selenium_fetch(url, session.headers.get("User-Agent"))
            except Exception:
                raise requests.exceptions.HTTPError(
                    f"403 Client Error for url {url}: site may be blocking automated requests (set SCRAPER_USER_AGENT or use a proxy)."
                )
            # Cache selenium result
            url_path.write_text(final, encoding="utf-8")
            temporary_path = html_path.with_suffix(".html.tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(html_path)
            return content, final
        raise
    url_path.write_text(response.url, encoding="utf-8")
    # Rename only after the complete HTML has been written.
    temporary_path = html_path.with_suffix(".html.tmp")
    temporary_path.write_bytes(response.content)
    temporary_path.replace(html_path)
    return response.content, response.url


def scrape_parishes(
    diocese: str,
    url: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Dispatch supported directories while preserving the shared output schema.

    Fort Wayne-South Bend has its own finder; other existing directory callers
    continue to use the eCatholic scraper.
    """
    hostname = (urlparse(url).hostname or "").lower().rstrip(".").removeprefix("www.")
    scraper = scrape_fwsb_parishes if hostname == "diocesefwsb.org" else scrape_ecatholic_parishes
    return scraper(diocese, url, cache_dir=cache_dir)


def scrape_ecatholic_parishes(
    diocese: str,
    url: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Scrape directory entries from the eCatholic parish-finder markup."""
    html, page_url = get_page(url, cache_dir=cache_dir)
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
        parishes.append(
            {
                "name": name_element.get_text(" ", strip=True),
                "address": (
                    address_element.get_text(" ", strip=True) if address_element else ""
                ),
                "site_url": urljoin(page_url, website["href"]) if website else "",
                "diocese": diocese,
                "category": (
                    category_name.get_text(" ", strip=True) if category_name else ""
                ),
                "parish_list_url": url,
            }
        )

    if not parishes:
        raise ValueError(f"No parish directory entries found for {diocese}: {url}")
    return pd.DataFrame(parishes, columns=PARISH_COLUMNS).drop_duplicates().reset_index(drop=True)


def scrape_fwsb_parishes(
    diocese: str,
    url: str = "https://diocesefwsb.org/find?show=parish",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Scrape parish cards and their website links from the FWSB finder.

    The server sends schools and ministries even with show=parish, so filter
    explicitly. Finder cards link to diocesan detail pages, not parish websites;
    read each cached/downloaded detail's Contact/Website field for site_url.
    Missing websites stay empty. Request or structural parsing failures stop
    the run, consistent with the eCatholic scraper.
    """
    html, page_url = get_page(url, cache_dir=cache_dir)
    soup = BeautifulSoup(html, "html.parser")
    entries = soup.select('.finder-entry[data-category="parish"]')
    if not entries:
        raise ValueError(f"No FWSB parish directory entries found for {diocese}: {url}")
    parishes = []
    seen_details = set()
    directory_host = (urlparse(page_url).hostname or "").lower().removeprefix("www.")
    for entry in entries:
        name_element = entry.select_one(".finder-entry-title")
        name = (entry.get("data-title") or (name_element.get_text(" ", strip=True) if name_element else "")).strip()
        if not name:
            raise ValueError(f"Missing directory entry name for {diocese}: {url}")
        link = entry.select_one("a.finder-entry-link[href]")
        href = (entry.get("data-url") or (link["href"] if link else "")).strip()
        detail_url = urljoin(page_url, href)
        detail_parts = urlparse(detail_url)
        if (
            not href or detail_parts.scheme not in {"http", "https"}
            or (detail_parts.hostname or "").lower().removeprefix("www.") != directory_host
            or not detail_parts.path.startswith("/parishes/")
        ):
            raise ValueError(f"Missing or invalid parish detail URL for {name}: {href}")
        if detail_url in seen_details:
            continue
        seen_details.add(detail_url)
        detail_html, final_detail_url = get_page(detail_url, cache_dir=cache_dir)
        detail = BeautifulSoup(detail_html, "html.parser")
        contact_heading = next((h for h in detail.find_all("h2") if h.get_text(strip=True).casefold() == "contact"), None)
        if not detail.find("h1") or contact_heading is None:
            raise ValueError(f"Missing parish detail content for {name}: {detail_url}")
        contact = contact_heading.parent
        website_heading = next((h for h in contact.find_all("h3") if h.get_text(strip=True).casefold() == "website"), None)
        website = website_heading.parent.select_one("a[href]") if website_heading else None
        site_url = urljoin(final_detail_url, website["href"].strip()) if website and website["href"].strip() else ""
        if site_url and (urlparse(site_url).scheme not in {"http", "https"} or not urlparse(site_url).hostname):
            raise ValueError(f"Invalid parish website URL for {name}: {site_url}")
        address = entry.get("data-address", "").strip()
        if not address:
            address_heading = next((h for h in contact.find_all("h3") if h.get_text(strip=True).casefold() == "address"), None)
            address_text = address_heading.parent.find("p") if address_heading else None
            address = address_text.get_text(", ", strip=True) if address_text else ""
        # Keep the church address ahead of explicitly labeled secondary offices/mail.
        address = re.split(r"\b(?:mailing|office)\s+address\s*:", address, maxsplit=1, flags=re.IGNORECASE)[0]
        address = ", ".join(part.strip() for part in address.split(",") if part.strip())
        parishes.append({
            "name": name, "address": address, "site_url": site_url,
            "diocese": diocese, "category": "Parish", "parish_list_url": url,
        })
        print(f"FWSB {len(parishes)}/{len(entries)}: {name}", flush=True)
    return pd.DataFrame(parishes, columns=PARISH_COLUMNS).drop_duplicates().reset_index(drop=True)


def _selenium_fetch(url: str, user_agent: str | None = None) -> Tuple[bytes, str]:
    """Attempt to fetch page with a headless Selenium browser.

    This is a best-effort, optional fallback. It requires `selenium` and a
    compatible webdriver (e.g., chromedriver) to be installed and available
    on PATH. The function raises an informative exception if Selenium is
    unavailable or fails.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options as ChromeOptions
    except Exception as e:
        raise RuntimeError(
            "Selenium is not available: install selenium and a webdriver"
        ) from e

    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    if user_agent:
        opts.add_argument(f"--user-agent={user_agent}")

    driver = webdriver.Chrome(options=opts)
    try:
        driver.set_page_load_timeout(60)
        driver.get(url)
        # Allow short time for JS to render
        time.sleep(2)
        content = driver.page_source
        final = driver.current_url
        return content.encode("utf-8"), final
    finally:
        try:
            driver.quit()
        except Exception:
            pass

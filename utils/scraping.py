import os
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
) -> tuple[bytes, str]:
    """Read cached HTML, or download and cache a successful response.

    Optionally record whether each fetch uses the cache or the web.
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
            parser.read()
        except Exception:
            # If robots.txt cannot be fetched/parsed, default to permissive behavior
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
    return pd.DataFrame(parishes).drop_duplicates().reset_index(drop=True)


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

"""Discover parish pages from sitemaps and homepage links."""

import gzip
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

from utils.scraping import get_page


def discover_site_pages(site_url: str, cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return deduplicated same-site pages and any discovery errors.

    Follow sitemap indexes recursively, but do not crawl individual page links.
    Successful downloads use the shared on-disk cache.
    """
    pages = {}
    errors = []

    def normalize(url):
        parts = urlsplit(urldefrag(url.strip())[0])
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))

    def host(url):
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")

    site_url = normalize(site_url)
    if not site_url:
        return [], [{"url": "", "error": "Missing or invalid parish site URL"}]
    allowed_hosts = {host(site_url)}

    def fetch(url):
        try:
            return get_page(url, cache_dir=cache_dir)
        except (requests.RequestException, OSError) as exc:
            errors.append({"url": url, "error": str(exc)})
            return None

    def add_page(url, source):
        url = normalize(url)
        if not url or host(url) not in allowed_hosts:
            return
        # Keep page URLs, excluding documents, images, and other assets.
        if Path(urlsplit(url).path).suffix.lower() in {
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
            ".ico", ".css", ".js", ".xml", ".zip", ".mp3", ".mp4",
            ".doc", ".docx", ".xls", ".xlsx", ".ics",
        }:
            return
        pages.setdefault(url, set()).add(source)

    # Check the conventional sitemap first, then robots.txt declarations.
    sitemap_url = urljoin(site_url, "/sitemap.xml")
    initial_sitemap = fetch(sitemap_url)
    robots = fetch(urljoin(site_url, "/robots.txt"))
    sitemap_urls = [sitemap_url]
    if robots:
        for line in robots[0].decode("utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "sitemap":
                sitemap_urls.append(urljoin(robots[1], value.strip()))

    # Resolve homepage redirects before filtering sitemap URLs by host.
    homepage = fetch(site_url)
    if homepage:
        allowed_hosts.add(host(homepage[1]))

    visited = set()

    def read_sitemap(url):
        url = normalize(url)
        if not url or url in visited:
            return
        visited.add(url)
        result = initial_sitemap if url == sitemap_url else fetch(url)
        if result is None:
            return
        content, final_url = result
        try:
            if content.startswith(b"\x1f\x8b"):
                content = gzip.decompress(content)
            root = ET.fromstring(content)
            kind = root.tag.rsplit("}", 1)[-1]
            if kind not in {"urlset", "sitemapindex"}:
                raise ValueError("Response is not a sitemap")
            # Only direct loc children: do not collect image/video extension URLs.
            for entry in root:
                for child in entry:
                    if child.tag.rsplit("}", 1)[-1] == "loc" and child.text:
                        target = urljoin(final_url, child.text.strip())
                        if kind == "sitemapindex":
                            read_sitemap(target)
                        else:
                            add_page(target, "sitemap")
        except (ET.ParseError, ValueError, OSError, EOFError) as exc:
            errors.append({"url": url, "error": str(exc)})

    for url in sitemap_urls:
        read_sitemap(url)

    if homepage:
        content, final_url = homepage
        add_page(final_url, "homepage")
        soup = BeautifulSoup(content, "html.parser")
        base = soup.select_one("base[href]")
        base_url = urljoin(final_url, base["href"]) if base else final_url
        for link in soup.select("a[href]"):
            add_page(urljoin(base_url, link["href"]), "homepage_link")

    return [
        {"page_url": url, "sources": ", ".join(sorted(sources))}
        for url, sources in sorted(pages.items())
    ], errors

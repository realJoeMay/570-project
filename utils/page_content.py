"""Extract readable page content using the shared HTML cache."""

from pathlib import Path

from bs4 import BeautifulSoup

from utils.scraping import get_page


def extract_page_content(
    page_url: str, cache_dir: str | Path | None = None,
) -> dict[str, str]:
    """Fetch a page and return its final URL, title, and normalized body text.

    Remove scripts, styles, embedded media, and explicitly hidden elements.
    Keep navigation and other visible text because they can name ministries.
    This extracts server-rendered HTML; it does not execute JavaScript.
    """
    html, final_url = get_page(page_url, cache_dir=cache_dir)
    soup = BeautifulSoup(html, "html.parser")
    title = " ".join(soup.title.get_text(" ", strip=True).split()) if soup.title else ""

    for element in soup.select(
        "script, style, noscript, template, svg, iframe, head, "
        "[hidden], [aria-hidden='true']"
    ):
        element.decompose()

    body = soup.body if soup.body is not None else soup
    clean_text = " ".join(body.get_text(" ", strip=True).split())
    return {"final_url": final_url, "page_title": title, "clean_text": clean_text}

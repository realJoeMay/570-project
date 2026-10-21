"""Extract reusable homepage links and apply editable social-link rules."""

from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd
from bs4 import BeautifulSoup

from utils.scraping import get_page


LINK_COLUMNS = ["homepage_url", "external_url", "external_host", "link_text", "link_count"]
HOMEPAGE_LINK_COLUMNS = ["homepage_url", "link_url", "link_host", "link_type", "link_text", "link_count"]


def read_homepage(site_url, cache_dir=None, *, cache_only=True):
    """Read either original or discovery-normalized homepage cache keys."""
    site_url = site_url.strip()
    parts = urlsplit(site_url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Invalid parish homepage URL: {site_url}")
    normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
    for candidate in dict.fromkeys([site_url, normalized]):
        try:
            return get_page(candidate, cache_dir=cache_dir, cache_only=True)
        except FileNotFoundError:
            pass
    if cache_only:
        raise FileNotFoundError(f"No cached homepage HTML for {site_url}")
    return get_page(normalized, cache_dir=cache_dir)


def _host(url):
    return (urlsplit(url).hostname or "").lower().rstrip(".").removeprefix("www.")


def extract_homepage_links(html, site_url, homepage_url):
    """Return distinct internal and external HTTP(S) hyperlink targets.

    Resolve links against the redirected homepage and HTML base URL. Original
    and redirected hosts (plus their subdomains) are internal. Preserve paths
    and queries; remove fragments and merge repeated targets, retaining labels.
    """
    soup = BeautifulSoup(html, "html.parser")
    base = soup.find("base", href=True)
    base_url = urljoin(homepage_url, base["href"]) if base else homepage_url
    internal_hosts = {_host(site_url), _host(homepage_url)} - {""}
    links = {}
    for anchor in soup.select("a[href], area[href]"):
        href = anchor["href"].strip()
        if not href or href.startswith("#"):
            continue
        try:
            parts = urlsplit(urljoin(base_url, href))
            host = _host(parts.geturl())
            if parts.scheme.lower() not in {"http", "https"} or not host:
                continue
            is_internal = any(host == internal or host.endswith("." + internal) for internal in internal_hosts)
            # Normalize host case and default ports; retain non-default ports.
            port = parts.port
            hostname = parts.hostname.lower().rstrip(".")
            if ":" in hostname:
                hostname = f"[{hostname}]"
            netloc = hostname if port in {None, 80 if parts.scheme == "http" else 443} else f"{hostname}:{port}"
            url = urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))
        except ValueError:
            continue
        label = " ".join((anchor.get_text(" ", strip=True) or anchor.get("aria-label") or anchor.get("title") or anchor.get("alt") or "").split())
        if not label:
            label = " ".join(image.get("alt", "") for image in anchor.find_all("img")).strip()
        record = links.setdefault(url, {
            "homepage_url": homepage_url, "link_url": url,
            "link_host": host, "link_type": "internal" if is_internal else "external",
            "link_text": [], "link_count": 0,
        })
        record["link_count"] += 1
        if label and label not in record["link_text"]:
            record["link_text"].append(label)
    return [{**row, "link_text": " | ".join(row["link_text"])} for row in links.values()]


def extract_external_links(html, site_url, homepage_url):
    """Keep the external-only output schema used by social-link extraction."""
    return [
        {"homepage_url": row["homepage_url"], "external_url": row["link_url"],
         "external_host": row["link_host"], "link_text": row["link_text"], "link_count": row["link_count"]}
        for row in extract_homepage_links(html, site_url, homepage_url)
        if row["link_type"] == "external"
    ]


def load_social_rules(path):
    """Load ordered domain/domain_path_prefix rules; first match wins."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"match_type", "pattern", "is_social", "service", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Social rules missing columns: {sorted(required - set(frame.columns))}")
    rules = []
    for index, row in frame.iterrows():
        rule = {key: row[key].strip() for key in required}
        rule["line"] = index + 2
        rule["pattern"] = rule["pattern"].lower().removeprefix("www.")
        if rule["match_type"] not in {"domain", "domain_path_prefix"}:
            raise ValueError(f"Unknown social rule type on line {index + 2}")
        domain, _, path_prefix = rule["pattern"].partition("/")
        if not domain or ":" in domain or " " in domain or (rule["match_type"] == "domain" and "/" in rule["pattern"]):
            raise ValueError(f"Invalid social rule pattern on line {index + 2}")
        if rule["match_type"] == "domain_path_prefix" and not path_prefix:
            raise ValueError(f"Missing rule path prefix on line {index + 2}")
        value = rule["is_social"].lower()
        if value not in {"true", "false"}:
            raise ValueError(f"is_social must be TRUE or FALSE on line {index + 2}")
        rule["is_social"] = value == "true"
        rules.append(rule)
    return rules


def classify_social_links(links, rules):
    """Annotate every external link, retaining non-social links for later work."""
    columns = ["is_social", "social_service", "social_category", "matched_social_rule"]
    result = links.drop(columns=columns, errors="ignore").copy()
    annotations = []
    for url in result["external_url"]:
        host = _host(url)
        path = urlsplit(url).path.lower()
        annotation = dict(is_social=False, social_service="", social_category="", matched_social_rule="")
        for rule in rules:
            domain, _, prefix = rule["pattern"].partition("/")
            if host != domain and not host.endswith("." + domain):
                continue
            if rule["match_type"] == "domain_path_prefix" and not path.startswith("/" + prefix):
                continue
            annotation.update(
                is_social=rule["is_social"], social_service=rule["service"],
                social_category=rule["category"], matched_social_rule=f"line {rule['line']}: {rule['pattern']}",
            )
            break
        annotations.append(annotation)
    for column in columns:
        result[column] = [row[column] for row in annotations]
    result["is_social"] = result["is_social"].astype(bool)
    return result

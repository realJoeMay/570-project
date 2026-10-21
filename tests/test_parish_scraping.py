"""Directory-specific parish parsing and shared scraper dispatch."""

import contextlib
import io
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from utils.scraping import (
    PARISH_COLUMNS, get_page, scrape_ecatholic_parishes,
    scrape_fwsb_parishes, scrape_parishes,
)


FINDER_URL = "https://diocesefwsb.org/find?show=parish"


class ParishScrapingTests(unittest.TestCase):
    def test_dispatch_preserves_existing_callers(self):
        for url, function in [
            (FINDER_URL, "scrape_fwsb_parishes"),
            ("https://www.diocesefwsb.org/find?show=parish", "scrape_fwsb_parishes"),
            ("https://mobarch.org/parishfinder", "scrape_ecatholic_parishes"),
        ]:
            with self.subTest(url=url), patch(f"utils.scraping.{function}") as scraper:
                self.assertIs(scrape_parishes("Test", url, cache_dir="cache"), scraper.return_value)
                scraper.assert_called_once_with("Test", url, cache_dir="cache")

    def test_ecatholic_schema_relative_links_and_exact_duplicates(self):
        entry = '''<li class="site"><div class="siteInfo"><div class="title">
        <span class="name">St. Example</span><span class="address">Town, AL</span>
        </div><div class="website"><a href="/parish">Website</a></div></div></li>'''
        html = f'<li class="category"><div class="categoryName">Parishes</div><ul>{entry}{entry}</ul></li>'
        with patch("utils.scraping.get_page", return_value=(html, "https://newdiocese.org/directory")):
            frame = scrape_ecatholic_parishes("Test", "https://old.org/list")
        self.assertEqual(frame.columns.tolist(), PARISH_COLUMNS)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].site_url, "https://newdiocese.org/parish")
        self.assertEqual(frame.iloc[0].category, "Parishes")
        self.assertEqual(frame.iloc[0].parish_list_url, "https://old.org/list")

    def test_fwsb_filters_categories_follows_details_and_retains_missing_website(self):
        directory = '''
        <div class="finder-entry" data-category="parish" data-title="St. One"
          data-url="/parishes/one" data-address="1 Main St., Town, IN 46000, , Mailing Address: PO Box 1, Town, IN 46000"></div>
        <div class="finder-entry" data-category="parish" data-title="St. One" data-url="/parishes/one"></div>
        <div class="finder-entry" data-category="parish"><a class="finder-entry-link" href="/parishes/two">
          <h3 class="finder-entry-title">St. Two</h3></a></div>
        <div class="finder-entry" data-category="school" data-title="School" data-url="/schools/one"></div>
        <div class="finder-entry" data-category="ministry" data-title="Ministry" data-url="/ministries/one"></div>
        '''
        first = '''<h1>St. One</h1><div><h2>Contact</h2><div><h3>Website</h3>
        <a href="https://parish-one.org/">Parish website</a></div></div>'''
        second = '''<h1>St. Two</h1><div><h2>Contact</h2><div><h3>Address</h3>
        <p>2 Church St.<br>Other Town, IN 46001</p></div></div>
        <footer><a href="https://diocesan-news.org/">Website</a></footer>'''
        responses = {
            FINDER_URL: (directory, FINDER_URL),
            "https://diocesefwsb.org/parishes/one": (first, "https://diocesefwsb.org/parishes/one"),
            "https://diocesefwsb.org/parishes/two": (second, "https://diocesefwsb.org/parishes/two"),
        }
        with patch("utils.scraping.get_page", side_effect=lambda url, **kwargs: responses[url]) as fetch:
            with contextlib.redirect_stdout(io.StringIO()):
                frame = scrape_fwsb_parishes("Test", cache_dir="cache")
        self.assertEqual(frame.columns.tolist(), PARISH_COLUMNS)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(frame.name.tolist(), ["St. One", "St. Two"])
        self.assertEqual(frame.site_url.tolist(), ["https://parish-one.org/", ""])
        self.assertEqual(frame.address.tolist(), ["1 Main St., Town, IN 46000", "2 Church St., Other Town, IN 46001"])
        self.assertTrue(frame.category.eq("Parish").all())

    def test_fwsb_rejects_missing_name_and_unexpected_markup(self):
        for html in ["<html></html>", '<div class="finder-entry" data-category="parish"></div>']:
            with self.subTest(html=html), patch("utils.scraping.get_page", return_value=(html, FINDER_URL)):
                with self.assertRaises(ValueError):
                    scrape_fwsb_parishes("Test")

    def test_fwsb_does_not_export_detail_url_as_parish_website(self):
        directory = '<div class="finder-entry" data-category="parish" data-title="Test" data-url="/parishes/test"></div>'
        with patch("utils.scraping.get_page", side_effect=[(directory, FINDER_URL), ("<h1>Error page</h1>", FINDER_URL)]):
            with self.assertRaisesRegex(ValueError, "Missing parish detail content"):
                scrape_fwsb_parishes("Test")


class RobotsUserAgentTests(unittest.TestCase):
    def test_robots_uses_the_page_session_and_still_enforces_disallow(self):
        for status, text, allowed in [
            (200, "User-agent: *\nDisallow: /cpresources/", True),
            (200, "User-agent: *\nDisallow: /parishes/", False),
            (403, "Forbidden", False),
        ]:
            with self.subTest(status=status, allowed=allowed), tempfile.TemporaryDirectory() as folder:
                robots = Mock(status_code=status, text=text)
                page = Mock(status_code=200, content=b"<html>Parish</html>", url="https://example.org/parishes/test")
                with patch("utils.scraping.requests.Session") as factory, patch("utils.scraping.time.sleep"), patch("utils.scraping._robots_parsers", {}, create=True), patch("utils.scraping._last_fetch_times", {}, create=True), patch.dict("os.environ", {"SCRAPER_IGNORE_ROBOTS": "0", "SCRAPER_USE_SELENIUM": "0"}):
                    session = factory.return_value
                    session.get.side_effect = [robots, page]
                    if allowed:
                        result = get_page(page.url, cache_dir=folder)
                        self.assertEqual(result, (page.content, page.url))
                        self.assertEqual(session.get.call_count, 2)
                    else:
                        with self.assertRaises(requests.HTTPError):
                            get_page(page.url, cache_dir=folder)
                        self.assertEqual(session.get.call_count, 1)
                    self.assertEqual(session.get.call_args_list[0].args[0], "https://example.org/robots.txt")


if __name__ == "__main__":
    unittest.main()
